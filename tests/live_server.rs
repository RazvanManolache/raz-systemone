//! Live server test: ephemeral port, Jev-style request/response, bad-scorer 400.
//! Needs Ollama up (embed scorer). Ignored by default.

#[tokio::test]
#[ignore]
async fn server_answers_systemone() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(systemone::server::serve_on(
        listener,
        systemone::server::ServerConfig {
            port: 0,
            default_scorer: "embed".to_string(),
            ollama_url: "http://localhost:11434".to_string(),
            embed_model: "nomic-embed-text".to_string(),
            llm_model: "unused".to_string(),
            nli_model: "unused".to_string(),
            jev_api_key: None,
            jev_model: "unused".to_string(),
        },
    ));
    let client = reqwest::Client::new();
    for _ in 0..50 {
        if client
            .get(format!("http://{addr}/healthz"))
            .send()
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let body = serde_json::json!({
        "state": include_str!("../examples/state.txt"),
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this",
                "criteria": {
                    "billing": "Payment or subscription issues",
                    "technical": "Bugs or integration problems",
                    "sales": "Pricing or account questions"
                }
            }
        }
    });
    let resp = client
        .post(format!("http://{addr}/v1/systemone"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["answers"]["department"]["choice"], "technical");
    assert_eq!(v["model"], "embed");

    let bad = serde_json::json!({
        "state": "x",
        "model": "nope",
        "questions": {"q": {"type": "noul", "instructions": "x"}},
    });
    let resp = client
        .post(format!("http://{addr}/v1/systemone"))
        .json(&bad)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    task.abort();
}
