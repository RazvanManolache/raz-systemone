//! HTTP server mimicking Jev's `POST /v1/systemone` (kept for compatibility): state + typed questions
//! in, typed answers out. Scorers live for the process lifetime, so the NLI
//! model loads once instead of per call.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::OnceCell;

use crate::ollama::OllamaClient;
use crate::{Answer, AskRequest, Question};

/// Which scorers the server offers.
pub const SCORERS: &[&str] = &["embed", "llm", "nli", "jev", "route"];

pub struct ServerConfig {
    pub port: u16,
    pub default_scorer: String,
    pub ollama_url: String,
    pub embed_model: String,
    pub llm_model: String,
    pub nli_model: String,
    pub jev_api_key: Option<String>,
    pub jev_model: String,
    pub route_spec: Option<String>,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    default_scorer: String,
    ollama: OllamaClient,
    embed_model: String,
    llm_model: String,
    nli_model: String,
    jev_api_key: Option<String>,
    jev_model: String,
    route_spec: Option<String>,
    /// Loaded lazily on first `nli` request (downloads + mmaps ~90MB).
    nli: OnceCell<crate::nli::NliScorer>,
    /// Built lazily on first `route` request (may load NLI checkpoints).
    route: OnceCell<crate::route::DynRouter>,
}

#[derive(Debug, serde::Deserialize)]
struct SystemOneRequest {
    state: String,
    #[serde(default)]
    model: Option<String>,
    questions: HashMap<String, Question>,
}

#[derive(Debug, serde::Serialize)]
struct SystemOneResponse {
    model: String,
    answers: HashMap<String, Answer>,
}

struct AppError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(serde_json::json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

async fn systemone(
    State(st): State<AppState>,
    Json(req): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, AppError> {
    let name = req
        .model
        .clone()
        .unwrap_or_else(|| st.inner.default_scorer.clone());
    let ask = AskRequest {
        state: req.state,
        questions: req.questions,
    };
    let answers = match name.as_str() {
        "embed" => {
            let s = crate::embed::EmbedScorer::new(
                st.inner.ollama.clone(),
                st.inner.embed_model.clone(),
            );
            crate::evaluate(&s, &ask).await?.answers
        }
        "llm" => {
            let s =
                crate::llm::LlmJudge::new(st.inner.ollama.clone(), st.inner.llm_model.clone());
            crate::evaluate(&s, &ask).await?.answers
        }
        "nli" => {
            let s = get_nli(&st).await?;
            crate::evaluate(s, &ask).await?.answers
        }
        "jev" => {
            let Some(key) = st.inner.jev_api_key.clone() else {
                return Err(AppError {
                    status: StatusCode::BAD_REQUEST,
                    message: "jev scorer not configured (TYPESAFE_API_KEY missing)".to_string(),
                });
            };
            let s = crate::jev::JevScorer::new(key, st.inner.jev_model.clone());
            crate::evaluate(&s, &ask).await?.answers
        }
        "route" => {
            let s = get_route(&st).await?;
            crate::evaluate(s, &ask).await?.answers
        }
        other => {
            return Err(AppError {
                status: StatusCode::BAD_REQUEST,
                message: format!("unknown scorer '{other}'; want one of {}", SCORERS.join(", ")),
            })
        }
    };
    Ok(Json(SystemOneResponse {
        model: name,
        answers,
    }))
}

/// Lazily built, process-lifetime router (spec fixed at startup).
async fn get_route(st: &AppState) -> Result<&crate::route::DynRouter, AppError> {
    let spec = st.inner.route_spec.clone().ok_or_else(|| AppError {
        status: StatusCode::BAD_REQUEST,
        message: "route scorer needs --route choice=<s>,score=<s>,noul=<s>".to_string(),
    })?;
    st.inner
        .route
        .get_or_try_init(|| {
            let inner = st.inner.clone();
            async move {
                tokio::task::spawn_blocking(move || {
                    let spec = crate::route::parse_route_spec(&spec)?;
                    crate::route::Backends {
                        ollama: inner.ollama.clone(),
                        embed_model: inner.embed_model.clone(),
                        llm_model: inner.llm_model.clone(),
                        nli_model: inner.nli_model.clone(),
                        jev_api_key: inner.jev_api_key.clone(),
                        jev_model: inner.jev_model.clone(),
                        temperature: 0.1,
                    }
                    .build_router(&spec)
                })
                .await
                .map_err(|e| anyhow::anyhow!("route build panicked: {e}"))?
            }
        })
        .await
        .map_err(AppError::from)
}

/// Lazily loaded, process-lifetime NLI scorer shared by `nli`/`ensemble`.
async fn get_nli(st: &AppState) -> Result<&crate::nli::NliScorer, AppError> {
    st.inner
        .nli
        .get_or_try_init(|| {
            let model = st.inner.nli_model.clone();
            async move {
                tokio::task::spawn_blocking(move || crate::nli::NliScorer::load(&model))
                    .await
                    .map_err(|e| anyhow::anyhow!("nli load panicked: {e}"))?
            }
        })
        .await
        .map_err(AppError::from)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn models_list(State(st): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "default": st.inner.default_scorer,
        "scorers": SCORERS,
        "backends": {
            "embed": st.inner.embed_model,
            "llm": st.inner.llm_model,
            "nli": st.inner.nli_model,
            "jev": st.inner.jev_model,
            "route": st.inner.route_spec,
        },
    }))
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/v1/models", get(models_list))
        .route("/healthz", get(healthz))
        .with_state(state)
}

/// Serve on an already-bound listener (used by tests).
pub async fn serve_on(
    listener: tokio::net::TcpListener,
    cfg: ServerConfig,
) -> anyhow::Result<()> {
    let state = AppState {
        inner: Arc::new(AppStateInner {
            default_scorer: cfg.default_scorer,
            ollama: OllamaClient::new(cfg.ollama_url),
            embed_model: cfg.embed_model,
            llm_model: cfg.llm_model,
            nli_model: cfg.nli_model,
            jev_api_key: cfg.jev_api_key,
            jev_model: cfg.jev_model,
            route_spec: cfg.route_spec,
            nli: OnceCell::new(),
            route: OnceCell::new(),
        }),
    };
    axum::serve(listener, router(state)).await?;
    Ok(())
}

/// Bind 127.0.0.1:`port` and serve forever.
pub async fn serve(cfg: ServerConfig) -> anyhow::Result<()> {
    if cfg.default_scorer == "route" {
        let spec = cfg
            .route_spec
            .as_deref()
            .context("default --scorer route needs --route choice=<s>,score=<s>,noul=<s>")?;
        crate::route::parse_route_spec(spec).context("invalid --route spec")?;
    }
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", cfg.port)).await?;
    eprintln!("raz listening on http://127.0.0.1:{}", cfg.port);
    serve_on(listener, cfg).await
}
