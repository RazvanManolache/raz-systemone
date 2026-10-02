//! Live tests against a local Ollama. Ignored by default (`cargo test` stays
//! hermetic); run with `cargo test -- --ignored` while Ollama is up.

use std::collections::HashMap;

use raz::ollama::OllamaClient;
use raz::{evaluate, AskRequest, Question};

fn sample_request() -> AskRequest {
    let questions: HashMap<String, Question> =
        serde_json::from_str(include_str!("../examples/questions.json")).unwrap();
    AskRequest {
        state: include_str!("../examples/state.txt").to_string(),
        questions,
    }
}

#[tokio::test]
#[ignore]
async fn embed_scorer_routes_to_technical() {
    let scorer = raz::embed::EmbedScorer::new(
        OllamaClient::new("http://localhost:11434"),
        "nomic-embed-text",
    );
    let resp = evaluate(&scorer, &sample_request()).await.unwrap();
    let dept = resp.answers.get("department").unwrap();
    let choice = match dept {
        raz::Answer::Choice { choice, .. } => choice,
        other => panic!("expected choice, got {other:?}"),
    };
    assert_eq!(choice, "technical");
    assert!(resp.answers.contains_key("frustration"));
    assert!(resp.answers.contains_key("is_urgent"));
}

#[tokio::test]
#[ignore]
async fn llm_judge_routes_to_technical() {
    let scorer = raz::llm::LlmJudge::new(
        OllamaClient::new("http://localhost:11434"),
        "llama3.2:3b-instruct-fp16",
    );
    let resp = evaluate(&scorer, &sample_request()).await.unwrap();
    let dept = resp.answers.get("department").unwrap();
    let choice = match dept {
        raz::Answer::Choice { choice, .. } => choice,
        other => panic!("expected choice, got {other:?}"),
    };
    assert_eq!(choice, "technical");
}
