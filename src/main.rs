//! `raz`: Jev-like typed decisions over local models.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, ValueEnum};

use raz::ollama::OllamaClient;
use raz::{AskRequest, Question};

#[derive(Debug, Parser)]
#[command(name = "raz", about = "Jev-like typed decisions over local models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Backend options shared by every subcommand that scores.
#[derive(Debug, Args)]
struct BackendArgs {
    /// Ollama base URL (embed/llm scorers).
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,
    /// Embedding model (embed scorer).
    #[arg(long, default_value = "nomic-embed-text")]
    embed_model: String,
    /// Chat model (llm scorer). phi4-abliterated is the measured best
    /// (holdout-validated); `judge-3b-4k` (local) is faster at lower accuracy.
    #[arg(long, default_value = "huihui_ai/phi4-abliterated")]
    llm_model: String,
    /// HF model id or local dir (nli scorer).
    #[arg(long, default_value = raz::nli::DEFAULT_NLI_MODEL)]
    nli_model: String,
    /// Jev model id (jev scorer; key from TYPESAFE_API_KEY or .env).
    #[arg(long, default_value = raz::jev::DEFAULT_JEV_MODEL)]
    jev_model: String,
    /// Softmax temperature over cosine scores (embed scorer).
    #[arg(long, default_value = "0.1")]
    temperature: f64,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    /// Answer questions about a state, print {"answers": {...}} as JSON.
    Ask {
        /// State text inline.
        #[arg(long, conflicts_with = "state_file")]
        state: Option<String>,
        /// File holding the state text ("-" = stdin).
        #[arg(long)]
        state_file: Option<PathBuf>,
        /// JSON file {"name": {"type": ...}} with the questions ("-" = stdin).
        #[arg(long)]
        questions: PathBuf,
        /// Which scorer to use.
        #[arg(long, value_enum, default_value = "embed")]
        scorer: ScorerKind,
        #[command(flatten)]
        backend: BackendArgs,
    },
    /// Serve Jev-compatible POST /v1/systemone.
    Serve {
        /// Port to bind on 127.0.0.1.
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Default scorer when the request omits "model".
        #[arg(long, value_enum, default_value = "nli")]
        scorer: ScorerKind,
        #[command(flatten)]
        backend: BackendArgs,
    },
    /// Score a labeled JSONL set: accuracy + calibration (ECE).
    Eval {
        /// Labeled cases, one JSON object per line (see tests/data).
        #[arg(long)]
        data: PathBuf,
        /// Which scorer to use.
        #[arg(long, value_enum, default_value = "nli")]
        scorer: ScorerKind,
        /// Stop after N cases (0 = all).
        #[arg(long, default_value = "0")]
        limit: usize,
        /// Write one JSON line per judgment (for offline fitting).
        #[arg(long)]
        dump: Option<PathBuf>,
        #[command(flatten)]
        backend: BackendArgs,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ScorerKind {
    Embed,
    Llm,
    Nli,
    Jev,
}

impl ScorerKind {
    fn as_str(self) -> &'static str {
        match self {
            ScorerKind::Embed => "embed",
            ScorerKind::Llm => "llm",
            ScorerKind::Nli => "nli",
            ScorerKind::Jev => "jev",
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let Cli { command } = Cli::parse();
    match command {
        Command::Ask {
            state,
            state_file,
            questions,
            scorer,
            backend,
        } => {
            let state = match (state, state_file) {
                (Some(s), None) => s,
                (None, Some(p)) => read_input(&p).context("read state file")?,
                _ => anyhow::bail!("pass exactly one of --state or --state-file"),
            };
            let raw = read_input(&questions).context("read questions file")?;
            let parsed: HashMap<String, Question> =
                serde_json::from_str(&raw).context("parse questions JSON")?;
            let request = AskRequest {
                state,
                questions: parsed,
            };
            let response = ask_with(scorer, &backend, &request).await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
            Ok(())
        }
        Command::Serve {
            port,
            scorer,
            backend,
        } => {
            raz::server::serve(raz::server::ServerConfig {
                port,
                default_scorer: scorer.as_str().to_string(),
                ollama_url: backend.ollama_url,
                embed_model: backend.embed_model,
                llm_model: backend.llm_model,
                nli_model: backend.nli_model,
                jev_api_key: raz::jev::read_api_key().ok(),
                jev_model: backend.jev_model,
            })
            .await
        }
        Command::Eval {
            data,
            scorer,
            limit,
            dump,
            backend,
        } => {
            let text = std::fs::read_to_string(&data).context("read eval data")?;
            let cases = raz::eval::load_cases(&text)?;
            let cases: Vec<_> = if limit > 0 {
                cases.into_iter().take(limit).collect()
            } else {
                cases
            };
            anyhow::ensure!(!cases.is_empty(), "no cases in {}", data.display());
            let scorer_impl = EvalScorer::load(scorer, &backend)?;
            let report = raz::eval::run(&scorer_impl, &cases).await?;
            if let Some(path) = dump {
                report.write_dump(&path)?;
            }
            println!("{report}");
            Ok(())
        }
    }
}

/// One loaded scorer usable by both ask and eval paths.
enum EvalScorer {
    Embed(raz::embed::EmbedScorer),
    Llm(raz::llm::LlmJudge),
    Nli(raz::nli::NliScorer),
    Jev(raz::jev::JevScorer),
}

impl EvalScorer {
    fn load(kind: ScorerKind, backend: &BackendArgs) -> Result<Self> {
        let client = OllamaClient::new(backend.ollama_url.clone());
        match kind {
            ScorerKind::Embed => {
                let mut s = raz::embed::EmbedScorer::new(client, backend.embed_model.clone());
                s.temperature = backend.temperature;
                Ok(Self::Embed(s))
            }
            ScorerKind::Llm => Ok(Self::Llm(raz::llm::LlmJudge::new(
                client,
                backend.llm_model.clone(),
            ))),
            ScorerKind::Nli => Ok(Self::Nli(raz::nli::NliScorer::load(&backend.nli_model)?)),
            ScorerKind::Jev => Ok(Self::Jev(raz::jev::JevScorer::from_env(
                backend.jev_model.clone(),
            )?)),
        }
    }
}

impl raz::Scorer for EvalScorer {
    async fn answer(
        &self,
        state: &str,
        question: &raz::Question,
    ) -> anyhow::Result<raz::Answer> {
        match self {
            Self::Embed(s) => s.answer(state, question).await,
            Self::Llm(s) => s.answer(state, question).await,
            Self::Nli(s) => s.answer(state, question).await,
            Self::Jev(s) => s.answer(state, question).await,
        }
    }
}

async fn ask_with(
    kind: ScorerKind,
    backend: &BackendArgs,
    request: &AskRequest,
) -> Result<raz::AskResponse> {
    Ok(raz::evaluate(&EvalScorer::load(kind, backend)?, request).await?)
}

fn read_input(path: &PathBuf) -> Result<String> {
    if path.as_os_str() == "-" {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    } else {
        Ok(std::fs::read_to_string(path)?)
    }
}
