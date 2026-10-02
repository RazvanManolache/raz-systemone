//! Minimal async client for the local Ollama API.

use anyhow::{Context, Result};
use serde::Deserialize;

/// Thin wrapper over `reqwest` with an Ollama base URL.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    /// Embed a batch of texts with `model` via `POST /api/embed`.
    pub async fn embed(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(Deserialize)]
        struct EmbedResponse {
            embeddings: Vec<Vec<f32>>,
        }

        let url = format!("{}/api/embed", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&serde_json::json!({ "model": model, "input": inputs }))
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .with_context(|| format!("embed model '{model}'"))?
            .json::<EmbedResponse>()
            .await
            .context("decode /api/embed response")?;
        anyhow::ensure!(
            resp.embeddings.len() == inputs.len(),
            "expected {} embeddings, got {}",
            inputs.len(),
            resp.embeddings.len()
        );
        Ok(resp.embeddings)
    }

    /// First generated position's top logprobs plus sampled text, via the
    /// OpenAI-compatible `POST /v1/chat/completions`.
    pub async fn first_token_logprobs(
        &self,
        model: &str,
        system: &str,
        user: &str,
        top_n: u32,
    ) -> Result<FirstToken> {
        #[derive(Deserialize)]
        struct ChatResponse {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: Message,
            logprobs: Option<Logprobs>,
        }
        #[derive(Deserialize)]
        struct Message {
            content: Option<String>,
        }
        #[derive(Deserialize)]
        struct Logprobs {
            content: Option<Vec<Position>>,
        }
        #[derive(Deserialize)]
        struct Position {
            top_logprobs: Vec<TopEntry>,
        }
        #[derive(Deserialize)]
        struct TopEntry {
            token: String,
            logprob: f64,
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&serde_json::json!({
                "model": model,
                "messages": [
                    { "role": "system", "content": system },
                    { "role": "user", "content": user },
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": true,
                "top_logprobs": top_n,
            }))
            .send()
            .await
            .with_context(|| format!("POST {url}"))?
            .error_for_status()
            .with_context(|| format!("chat model '{model}'"))?
            .json::<ChatResponse>()
            .await
            .context("decode /v1/chat/completions response")?;

        let choice = resp.choices.into_iter().next().context("no choices")?;
        let position = choice
            .logprobs
            .and_then(|l| l.content)
            .and_then(|mut c| c.pop())
            .context("no logprobs in response (model/endpoint may not support them)")?;
        Ok(FirstToken {
            sampled: choice.message.content.unwrap_or_default(),
            top: position
                .top_logprobs
                .into_iter()
                .map(|e| (e.token, e.logprob))
                .collect(),
        })
    }
}

/// What the model thought about its first output token.
#[derive(Debug, Clone)]
pub struct FirstToken {
    /// The sampled token text (temperature 0, so the argmax).
    pub sampled: String,
    /// (token, logprob) candidates, most likely first.
    pub top: Vec<(String, f64)>,
}
