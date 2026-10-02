//! Stronger scorer: a small chat model answers a constrained single-token
//! question, and probabilities come from that position's logprobs.
//!
//! Each question is rendered with single-token labels (letters for choice,
//! digits for score, Yes/No for noul). The model's top logprobs over those
//! labels are renormalized into a distribution. Unlike the embedding baseline,
//! this captures negation and simple inference, at the cost of a generation
//! call per question.

use std::collections::HashMap;

use anyhow::Result;

use crate::math::{argmax, softmax};
use crate::ollama::FirstToken;
use crate::{ollama::OllamaClient, Answer, Question, Scorer};

/// LLM-judge scorer using first-token logprobs.
pub struct LlmJudge {
    client: OllamaClient,
    model: String,
    /// How many top logprobs to request; must cover the label set.
    pub top_n: u32,
}

impl LlmJudge {
    pub fn new(client: OllamaClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            top_n: 20,
        }
    }

    async fn distribution(&self, user: &str, labels: &[String]) -> Result<Vec<f64>> {
        let first = self
            .client
            .first_token_logprobs(
                &self.model,
                "Answer with exactly one token: the label only, nothing else.",
                user,
                self.top_n,
            )
            .await?;
        Ok(distribution_from_logprobs(&first, labels)?)
    }
}

impl Scorer for LlmJudge {
    async fn answer(&self, state: &str, question: &Question) -> Result<Answer> {
        match question {
            Question::Choice {
                instructions,
                criteria,
            } => {
                anyhow::ensure!(!criteria.is_empty(), "choice needs at least one option");
                anyhow::ensure!(
                    criteria.len() <= 26,
                    "choice supports at most 26 options (A-Z labels)"
                );
                let mut keys: Vec<&String> = criteria.keys().collect();
                keys.sort();
                let labels: Vec<String> = (0..keys.len())
                    .map(|i| ((b'A' + i as u8) as char).to_string())
                    .collect();
                let mut user = format!("Text: {state}\nQuestion: {instructions}\nOptions:\n");
                for (label, key) in labels.iter().zip(keys.iter()) {
                    user.push_str(&format!("{label}. {key}: {}\n", criteria[*key]));
                }
                user.push_str("Reply with only the letter of the best option.");
                let probs = self.distribution(&user, &labels).await?;
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
            Question::Score {
                instructions,
                criteria,
            } => {
                anyhow::ensure!(
                    !criteria.is_empty() && criteria.len() <= 10,
                    "score needs 1-10 levels (single-digit labels)"
                );
                let labels: Vec<String> = (0..criteria.len()).map(|i| i.to_string()).collect();
                let mut user = format!("Text: {state}\nQuestion: {instructions}\nLevels:\n");
                for (i, c) in criteria.iter().enumerate() {
                    user.push_str(&format!("{i}. {c}\n"));
                }
                user.push_str("Reply with only the number of the best level.");
                let probs = self.distribution(&user, &labels).await?;
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
                let labels = vec!["Yes".to_string(), "No".to_string()];
                let user = format!(
                    "Text: {state}\nStatement: {instructions}\nIs the statement true of the text? Example: Text: \"No rush, whenever you can.\" Statement: \"The message is urgent\" Answer: No\nReply with only Yes or No."
                );
                let probs = self.distribution(&user, &labels).await?;
                Ok(Answer::Noul { noul: probs[0] })
            }
        }
    }
}

/// Renormalize first-token logprobs over our label set.
///
/// Tokenizers often emit labels with a leading space (` Yes`), so matching
/// trims whitespace and ignores case. A label missing from the top list gets a
/// floor logprob instead of zero mass; if *no* label appears, this errors with
/// the observed tokens rather than inventing a distribution.
pub fn distribution_from_logprobs(first: &FirstToken, labels: &[String]) -> Result<Vec<f64>> {
    let mut found: HashMap<usize, f64> = HashMap::new();
    for (token, logprob) in &first.top {
        let norm = token.trim().to_lowercase();
        for (i, label) in labels.iter().enumerate() {
            if norm == label.to_lowercase() {
                found
                    .entry(i)
                    .and_modify(|v| *v = v.max(*logprob))
                    .or_insert(*logprob);
            }
        }
    }
    anyhow::ensure!(
        !found.is_empty(),
        "none of the labels {:?} appeared in top tokens (sampled {:?}; top: {:?})",
        labels,
        first.sampled,
        first.top.iter().map(|(t, _)| t).collect::<Vec<_>>()
    );
    let floor = found.values().cloned().fold(f64::INFINITY, f64::min) - 5.0;
    let logits: Vec<f64> = (0..labels.len())
        .map(|i| found.get(&i).cloned().unwrap_or(floor))
        .collect();
    Ok(softmax(&logits, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(top: Vec<(&str, f64)>) -> FirstToken {
        FirstToken {
            sampled: top[0].0.to_string(),
            top: top
                .into_iter()
                .map(|(t, l)| (t.to_string(), l))
                .collect(),
        }
    }

    #[test]
    fn matches_spaced_and_cased_variants() {
        let f = first(vec![(" Yes", -0.2), ("No", -2.0), ("maybe", -5.0)]);
        let p = distribution_from_logprobs(&f, &["Yes".into(), "No".into()]).unwrap();
        assert!((p[0] + p[1] - 1.0).abs() < 1e-9);
        assert!(p[0] > 0.8);
    }

    #[test]
    fn missing_label_gets_floor_mass() {
        let f = first(vec![("A", -0.1), ("B", -3.0)]);
        let labels = vec!["A".into(), "B".into(), "C".into()];
        let p = distribution_from_logprobs(&f, &labels).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(p[2] < 0.01 && p[2] > 0.0);
        assert!(p[0] > p[1]);
    }

    #[test]
    fn errors_when_no_label_visible() {
        let f = first(vec![("Sure", -0.1), ("thing", -1.0)]);
        let err = distribution_from_logprobs(&f, &["A".into()]).unwrap_err();
        assert!(err.to_string().contains("none of the labels"));
    }
}
