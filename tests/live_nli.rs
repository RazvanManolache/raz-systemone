//! Live NLI test: downloads the ~90MB checkpoint from HF Hub on first run
//! (cached after). Ignored by default; run with `cargo test -- --ignored`.

use std::collections::HashMap;

use systemone::{evaluate, AskRequest, Question};

fn request_for(state: &str) -> AskRequest {
    let questions: HashMap<String, Question> =
        serde_json::from_str(include_str!("../examples/questions.json")).unwrap();
    AskRequest {
        state: state.to_string(),
        questions: questions
            .into_iter()
            .filter(|(k, _)| k == "department" || k == "is_urgent")
            .collect(),
    }
}

#[tokio::test]
#[ignore]
async fn nli_routes_ticket_and_rejects_negation() {
    let scorer = systemone::nli::NliScorer::load(systemone::nli::DEFAULT_NLI_MODEL).unwrap();
    let resp = evaluate(
        &scorer,
        &request_for(include_str!("../examples/state.txt")),
    )
    .await
    .unwrap();
    match resp.answers.get("department").unwrap() {
        systemone::Answer::Choice { choice, .. } => assert_eq!(choice, "technical"),
        other => panic!("expected choice, got {other:?}"),
    }
    match resp.answers.get("is_urgent").unwrap() {
        systemone::Answer::Noul { noul } => assert!(*noul > 0.7, "urgent ticket: {noul}"),
        other => panic!("expected noul, got {other:?}"),
    }

    let resp = evaluate(
        &scorer,
        &request_for("There is no urgency here. Take your time, no rush at all."),
    )
    .await
    .unwrap();
    match resp.answers.get("is_urgent").unwrap() {
        systemone::Answer::Noul { noul } => assert!(*noul < 0.3, "negation: {noul}"),
        other => panic!("expected noul, got {other:?}"),
    }
}
