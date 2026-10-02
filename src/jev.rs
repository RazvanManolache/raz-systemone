//! Scorer backed by the real Jev API (TypeSafe AI): the reference our local
//! scorers are measured against.

use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::{Answer, Question, Scorer};

pub const DEFAULT_JEV_MODEL: &str = "jev-latest";
const ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";

pub struct JevScorer {
    http: reqwest::Client,
    api_key: String,
    model: String,
}

impl JevScorer {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        Ok(Self::new(read_api_key()?, model))
    }
}

/// `TYPESAFE_API_KEY` from the environment, falling back to a `.env` file.
pub fn read_api_key() -> Result<String> {
    if let Ok(k) = std::env::var("TYPESAFE_API_KEY") {
        if !k.trim().is_empty() {
            return Ok(k.trim().to_string());
        }
    }
    let text = std::fs::read_to_string(".env").context("set TYPESAFE_API_KEY or add it to .env")?;
    for line in text.lines() {
        if let Some(v) = line.trim().strip_prefix("TYPESAFE_API_KEY=") {
            let v = v.trim().trim_matches('"').to_string();
            if !v.is_empty() {
                return Ok(v);
            }
        }
    }
    anyhow::bail!("TYPESAFE_API_KEY not found in env or .env")
}

impl Scorer for JevScorer {
    async fn answer(&self, state: &str, question: &Question) -> Result<Answer> {
        // One question per call: Jev evaluates questions in isolation anyway.
        let mut questions = HashMap::new();
        questions.insert("q".to_string(), question.clone());
        let resp = self
            .http
            .post(ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "state": state,
                "model": self.model,
                "questions": questions,
            }))
            .send()
            .await
            .context("POST api.typesafe.ai/v1/systemone")?;
        if !resp.status().is_success() {
            let (status, text) = (resp.status(), resp.text().await.unwrap_or_default());
            anyhow::bail!("jev api {status}: {text}");
        }
        let decoded: JevResponse = resp.json().await.context("decode jev response")?;
        decoded.answers.into_iter().next().map(|(_, a)| a).context("jev returned no answer")
    }
}

#[derive(serde::Deserialize)]
struct JevResponse {
    answers: HashMap<String, Answer>,
}
