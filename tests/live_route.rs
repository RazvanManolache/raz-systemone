//! Live router test: builds a `DynRouter` over the stock NLI checkpoint
//! (~90MB from HF Hub on first run, cached after) and checks each leg
//! answers its own question type. Ignored by default.

use std::collections::HashMap;

use raz::{evaluate, route, AskRequest, Question};

#[tokio::test]
#[ignore]
async fn router_dispatches_each_leg() {
    let spec = route::parse_route_spec("choice=nli,score=nli,noul=nli").unwrap();
    let backends = route::Backends {
        ollama: raz::ollama::OllamaClient::new("http://localhost:11434".to_string()),
        embed_model: "nomic-embed-text".to_string(),
        llm_model: "unused".to_string(),
        nli_model: raz::nli::DEFAULT_NLI_MODEL.to_string(),
        jev_api_key: None,
        jev_model: "unused".to_string(),
        temperature: 0.1,
    };
    let router = backends.build_router(&spec).unwrap();
    let questions: HashMap<String, Question> =
        serde_json::from_str(include_str!("../examples/questions.json")).unwrap();
    let resp = evaluate(
        &router,
        &AskRequest {
            state: include_str!("../examples/state.txt").to_string(),
            questions,
        },
    )
    .await
    .unwrap();
    assert!(resp.answers.contains_key("department"));
    assert!(resp.answers.contains_key("frustration"));
    assert!(resp.answers.contains_key("is_urgent"));
}
