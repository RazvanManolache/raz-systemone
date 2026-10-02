//! Cross-encoder NLI scorer: the state and each hypothesis are read *together*
//! by a small DeBERTa-v3 model fine-tuned on MNLI/SNLI, and the entailment
//! probability becomes the score. Unlike bi-encoder similarity, this sees
//! negation and inference; unlike the LLM judge, it needs no generation.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{Config, DebertaV2SeqClassificationModel};
use hf_hub::{split_id, HFClientSync};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use crate::math::{argmax, softmax};
use crate::{Answer, Question, Scorer};

/// Small English NLI cross-encoder (DeBERTa-v3 arch, ~90MB).
pub const DEFAULT_NLI_MODEL: &str = "cross-encoder/nli-deberta-v3-xsmall";

/// NLI cross-encoder scorer. Downloads the checkpoint from HF Hub on first load
/// (cached under `~/.cache/huggingface`), runs on CPU.
#[derive(Clone)]
pub struct NliScorer {
    inner: std::sync::Arc<Inner>,
}

struct Inner {
    model: DebertaV2SeqClassificationModel,
    tokenizer: Tokenizer,
    entail_idx: usize,
    contra_idx: usize,
    id2label: std::collections::HashMap<u32, String>,
    device: Device,
}

impl NliScorer {
    pub fn load(model_id: &str) -> Result<Self> {
        // Local dir (e.g. a training/run output) or HF Hub id.
        let (config_bytes, weights, tok_path) = if std::path::Path::new(model_id).is_dir() {
            let dir = std::path::Path::new(model_id);
            for f in ["config.json", "model.safetensors", "tokenizer.json"] {
                anyhow::ensure!(dir.join(f).is_file(), "local NLI dir lacks {f}");
            }
            (
                std::fs::read(dir.join("config.json")).context("read config.json")?,
                dir.join("model.safetensors"),
                dir.join("tokenizer.json"),
            )
        } else {
            let client = HFClientSync::new().context("connect to huggingface hub")?;
            let (owner, name) = split_id(model_id);
            let repo = client.model(owner, name);
            let file = |n: &str| repo.download_file().filename(n).send();
            (
                std::fs::read(file("config.json")?).context("read config.json")?,
                file("model.safetensors")?,
                file("tokenizer.json")?,
            )
        };
        let config: Config = serde_json::from_slice(&config_bytes).context("parse config")?;

        let entail_idx = find_label(&config, "entail").context("no entailment label in id2label")?;
        let contra_idx = find_label(&config, "contra").context("no contradiction label in id2label")?;
        let id2label = config.id2label.clone().unwrap_or_default();

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)
                .context("mmap weights")?
        };
        // Encoder lives under `deberta.*`; the loader uses `vb.root()` for
        // `pooler.*` / `classifier.*`, so pass the prefixed builder.
        let model = DebertaV2SeqClassificationModel::load(vb.pp("deberta"), &config, None)
            .context("load deberta weights")?;

        let mut tokenizer =
            Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| anyhow::anyhow!("truncation: {e}"))?;
        let pad_id = config
            .pad_token_id
            .map(|id| id as u32)
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .context("no pad token for batching")?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        Ok(Self {
            inner: std::sync::Arc::new(Inner {
                model,
                tokenizer,
                entail_idx,
                contra_idx,
                id2label,
                device,
            }),
        })
    }

    /// Full 3-class distribution for one pair, as (label, prob) in id order.
    pub fn pair_scores(&self, premise: &str, hypothesis: &str) -> Result<Vec<(String, f64)>> {
        let (labels, probs) = Self::forward(&self.inner, premise, hypothesis)?;
        Ok(labels.into_iter().zip(probs.into_iter()).collect())
    }

    fn forward(inner: &Inner, premise: &str, hypothesis: &str) -> Result<(Vec<String>, Vec<f64>)> {
        let enc = inner
            .tokenizer
            .encode((premise, hypothesis), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let input_ids = Tensor::new(enc.get_ids(), &inner.device)?.unsqueeze(0)?;
        let type_ids = Tensor::new(enc.get_type_ids(), &inner.device)?.unsqueeze(0)?;
        let logits = inner
            .model
            .forward(&input_ids, Some(type_ids), None)
            .context("nli forward")?;
        let logits: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let probs = softmax(
            &logits.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            1.0,
        );
        let mut labels: Vec<(u32, String)> = inner
            .id2label
            .iter()
            .map(|(id, name)| (*id, name.clone()))
            .collect();
        labels.sort_by_key(|(id, _)| *id);
        Ok((
            labels.iter().map(|(_, n)| n.clone()).collect(),
            probs,
        ))
    }

    /// Score several hypotheses against one state in a single forward pass.
    fn score_all(
        inner: &Inner,
        state: &str,
        hyps: &[String],
        kind: ScoreFn,
    ) -> Result<Vec<f64>> {
        let pairs: Vec<(&str, &str)> = hyps.iter().map(|h| (state, h.as_str())).collect();
        let rows = Self::forward_batch(inner, &pairs)?;
        Ok(rows
            .iter()
            .map(|(e, c)| match kind {
                ScoreFn::Entail => *e,
                ScoreFn::Contrast => contrast_score(*e, *c),
            })
            .collect())
    }

    /// (entail, contra) for a batch of pairs in ONE forward pass: pad to the
    /// longest pair and run once instead of once per pair.
    fn forward_batch(inner: &Inner, pairs: &[(&str, &str)]) -> Result<Vec<(f64, f64)>> {
        anyhow::ensure!(!pairs.is_empty(), "empty batch");
        let encs = inner
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize batch: {e}"))?;
        let (b, t) = (encs.len(), encs[0].len());
        let stack = |f: fn(&tokenizers::Encoding) -> &[u32]| -> Result<Tensor> {
            let flat: Vec<u32> = encs.iter().flat_map(|e| f(e).to_vec()).collect();
            Ok(Tensor::new(flat, &inner.device)?.reshape((b, t))?)
        };
        let input_ids = stack(tokenizers::Encoding::get_ids)?;
        let type_ids = stack(tokenizers::Encoding::get_type_ids)?;
        let mask = stack(tokenizers::Encoding::get_attention_mask)?;
        let logits = inner
            .model
            .forward(&input_ids, Some(type_ids), Some(mask))
            .context("nli batch forward")?;
        let rows: Vec<Vec<f32>> = logits.to_vec2()?;
        Ok(rows
            .iter()
            .map(|logits| {
                let probs = softmax(
                    &logits.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                    1.0,
                );
                (probs[inner.entail_idx], probs[inner.contra_idx])
            })
            .collect())
    }
}

impl Scorer for NliScorer {
    async fn answer(&self, state: &str, question: &Question) -> Result<Answer> {
        // CPU-bound: run off the async workers. Forwards share the model
        // without locking (candle inference is read-only over Arc tensors).
        let inner = self.inner.clone();
        let state = state.to_string();
        let question = question.clone();
        tokio::task::spawn_blocking(move || {
            match &question {
                Question::Choice { criteria, .. } => {
                    anyhow::ensure!(!criteria.is_empty(), "choice needs at least one option");
                    let mut keys: Vec<&String> = criteria.keys().collect();
                    keys.sort();
                    // Zero-shot template: bare fragments read as neutral, so
                    // phrase each option as a full claim. Key words excluded:
                    // they bias entailment ("billing:" boosts billing).
                    let hyps: Vec<String> = keys
                        .iter()
                        .map(|k| format!("This text is about {}.", criteria[*k]))
                        .collect();
                    let scores = Self::score_all(&inner, &state, &hyps, ScoreFn::Entail)?;
                    let sum: f64 = scores.iter().sum();
                    let probs: Vec<f64> = scores.iter().map(|s| s / sum.max(1e-9)).collect();
                    let best = argmax(&probs);
                    Ok(Answer::Choice {
                        choice: keys[best].clone(),
                        probabilities: keys
                            .iter()
                            .zip(probs.iter())
                            .map(|(k, p)| ((*k).clone(), *p))
                            .collect(),
                        confidence: probs[best],
                    })
                }
                Question::Score { criteria, .. } => {
                    anyhow::ensure!(!criteria.is_empty(), "score needs at least one level");
                    // Tone hypotheses carry almost no entailment mass, so rank
                    // by contrast (entail vs contradiction) instead.
                    let scores = Self::score_all(&inner, &state, criteria, ScoreFn::Contrast)?;
                    let sum: f64 = scores.iter().sum();
                    let probs: Vec<f64> = scores.iter().map(|s| s / sum.max(1e-9)).collect();
                    let best = argmax(&probs);
                    Ok(Answer::Score {
                        score: best as f64,
                        probabilities: probs
                            .iter()
                            .enumerate()
                            .map(|(i, p)| (i.to_string(), *p))
                            .collect(),
                        confidence: probs[best],
                        legend: criteria
                            .iter()
                            .enumerate()
                            .map(|(i, c)| (i.to_string(), c.clone()))
                            .collect(),
                    })
                }
                Question::Noul { instructions } => {
                    // Contrast ignores neutral mass so meta-statements
                    // ("the message conveys urgency") still resolve.
                    let rows =
                        Self::forward_batch(&inner, &[(state.as_str(), instructions.as_str())])?;
                    Ok(Answer::Noul {
                        noul: contrast_score(rows[0].0, rows[0].1),
                    })
                }
            }
        })
        .await?
    }
}

#[derive(Clone, Copy)]
enum ScoreFn {
    Entail,
    Contrast,
}

/// Entail-vs-contradiction contrast; 0.5 when neither fires (no signal).
fn contrast_score(entail: f64, contra: f64) -> f64 {
    if entail + contra < 1e-6 {
        0.5
    } else {
        entail / (entail + contra)
    }
}

fn find_label(config: &Config, want: &str) -> Option<usize> {
    config
        .id2label
        .as_ref()?
        .iter()
        .find(|(_, name)| name.to_lowercase().contains(want))
        .map(|(id, _)| *id as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_ignores_neutral_mass() {
        assert!((contrast_score(0.0112, 0.0016) - 0.875).abs() < 0.01);
        assert!(contrast_score(0.0004, 0.95) < 0.01);
        assert!(contrast_score(0.99, 0.001) > 0.99);
    }

    #[test]
    fn contrast_abstains_without_signal() {
        assert_eq!(contrast_score(0.0, 0.0), 0.5);
    }
}
