//! Live Jev test: hits the real TypeSafe API. Needs TYPESAFE_API_KEY (env or
//! .env); passes trivially without it. Ignored by default.

use std::collections::HashMap;

use systemone::{evaluate, AskRequest, Question};

#[tokio::test]
#[ignore]
async fn jev_routes_sample_ticket() {
    let key = match systemone::jev::read_api_key() {
        Ok(k) => k,
        Err(_) => {
            eprintln!("skipping: TYPESAFE_API_KEY not set");
            return;
        }
    };
    let questions: HashMap<String, Question> =
        serde_json::from_str(include_str!("../examples/questions.json")).unwrap();
    let resp = evaluate(
        &systemone::jev::JevScorer::new(key, systemone::jev::DEFAULT_JEV_MODEL),
        &AskRequest {
            state: include_str!("../examples/state.txt").to_string(),
            questions,
        },
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
}
