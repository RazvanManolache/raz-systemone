//! Core types for a Jev-like decision API: typed questions in, calibrated answers out.
//!
//! Mirrors the three Jev primitives (`choice`, `score`, `noul`) so scorers are
//! interchangeable behind [`Scorer`].

pub mod embed;
pub mod eval;
pub mod jev;
pub mod llm;
pub mod math;
pub mod nli;
pub mod ollama;
pub mod route;
pub mod server;

use std::collections::HashMap;

/// One typed question about the state, Jev-style.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Pick one key from `criteria` (key -> description).
    Choice {
        instructions: String,
        criteria: HashMap<String, String>,
    },
    /// Pick one rubric level; levels are ordered as given.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    /// Probability that the statement holds of the state.
    Noul { instructions: String },
}

/// One typed answer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: HashMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        probabilities: HashMap<String, f64>,
        confidence: f64,
        legend: HashMap<String, String>,
    },
    Noul {
        noul: f64,
    },
}

/// Full request: state text plus a map of named questions.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AskRequest {
    pub state: String,
    pub questions: HashMap<String, Question>,
}

/// Full response: one named answer per question.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AskResponse {
    pub answers: HashMap<String, Answer>,
}

/// Something that can answer one typed question about a state.
pub trait Scorer: Send + Sync {
    fn answer(
        &self,
        state: &str,
        question: &Question,
    ) -> impl std::future::Future<Output = anyhow::Result<Answer>> + Send;
}

/// Evaluate every question against the same state, concurrently.
pub async fn evaluate<S: Scorer>(
    scorer: &S,
    request: &AskRequest,
) -> anyhow::Result<AskResponse> {
    use futures::future::join_all;

    let jobs: Vec<_> = request
        .questions
        .iter()
        .map(|(name, q)| async { anyhow::Ok((name.clone(), scorer.answer(&request.state, q).await?)) })
        .collect();
    let mut answers = HashMap::new();
    for done in join_all(jobs).await {
        let (name, answer) = done?;
        answers.insert(name, answer);
    }
    Ok(AskResponse { answers })
}
