//! Baseline scorer: bi-encoder cosine similarity + softmax.
//!
//! Each option is embedded once, the state is embedded once, and probabilities
//! come from `softmax(cosine / temperature)`. Fast and dependency-light, but
//! uncalibrated: embeddings measure topical similarity, not truth, so negation
//! and inference-heavy questions suffer. That gap is what [`crate::llm`] covers.

use anyhow::{Context, Result};

use crate::math::{argmax, cosine, sigmoid, softmax};
use crate::{ollama::OllamaClient, Answer, Question, Scorer};

/// Embeddings-based scorer (your original hypothesis, as a baseline).
pub struct EmbedScorer {
    client: OllamaClient,
    model: String,
    /// Softmax temperature over cosine scores. Cosines cluster in a narrow
    /// band, so this is small (default 0.1) to get a non-uniform distribution.
    pub temperature: f64,
    /// Gain mapping cosine -> noul probability via `sigmoid(gain * cosine)`.
    pub noul_gain: f64,
}

impl EmbedScorer {
    pub fn new(client: OllamaClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            temperature: 0.1,
            noul_gain: 10.0,
        }
    }

    async fn similarities(&self, state: &str, docs: &[String]) -> Result<Vec<f64>> {
        let mut inputs = Vec::with_capacity(docs.len() + 1);
        inputs.push(state.to_string());
        inputs.extend(docs.iter().cloned());
        let embs = self.client.embed(&self.model, &inputs).await?;
        let (query, docs) = embs.split_first().context("empty embedding batch")?;
        Ok(docs.iter().map(|d| cosine(query, d)).collect())
    }
}

impl Scorer for EmbedScorer {
    async fn answer(&self, state: &str, question: &Question) -> Result<Answer> {
        match question {
            Question::Choice {
                instructions: _,
                criteria,
            } => {
                anyhow::ensure!(!criteria.is_empty(), "choice needs at least one option");
                let mut keys: Vec<&String> = criteria.keys().collect();
                keys.sort();
                let docs: Vec<String> = keys
                    .iter()
                    .map(|k| format!("{}: {}", k, criteria[*k]))
                    .collect();
                let sims = self.similarities(state, &docs).await?;
                let probs = softmax(&sims, self.temperature);
                let winner = keys[argmax(&probs)].to_string();
                let confidence = probs[argmax(&probs)];
                let probabilities = keys
                    .iter()
                    .zip(probs.iter())
                    .map(|(k, p)| ((*k).clone(), *p))
                    .collect();
                Ok(Answer::Choice {
                    choice: winner,
                    probabilities,
                    confidence,
                })
            }
            Question::Score {
                instructions: _,
                criteria,
            } => {
                anyhow::ensure!(!criteria.is_empty(), "score needs at least one level");
                let sims = self.similarities(state, criteria).await?;
                let probs = softmax(&sims, self.temperature);
                let best = argmax(&probs);
                let probabilities = probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i.to_string(), *p))
                    .collect();
                let legend = criteria
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i.to_string(), c.clone()))
                    .collect();
                Ok(Answer::Score {
                    score: best as f64,
                    probabilities,
                    confidence: probs[best],
                    legend,
                })
            }
            Question::Noul { instructions } => {
                let sims = self
                    .similarities(state, &vec![instructions.clone()])
                    .await?;
                Ok(Answer::Noul {
                    noul: sigmoid(self.noul_gain * sims[0]),
                })
            }
        }
    }
}
