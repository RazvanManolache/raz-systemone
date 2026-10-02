//! Per-question-type router: `choice`, `score`, and `noul` questions are each
//! answered by their own scorer, e.g. `--route choice=nli:./v7,score=jev,noul=nli:./v5`.
//! Each leg is `name[:model]`; a bare name reuses the backend defaults.

use anyhow::{Context, Result};

use crate::{Answer, Question, Scorer};

/// One router leg: a scorer name plus an optional model override.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leg {
    pub name: String,
    pub model: Option<String>,
}

/// Parsed `--route` spec. All three legs are required (explicit beats magic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteSpec {
    pub choice: Leg,
    pub score: Leg,
    pub noul: Leg,
}

fn parse_leg(s: &str) -> Result<Leg> {
    let s = s.trim();
    anyhow::ensure!(!s.is_empty(), "empty leg in route spec");
    let (name, model) = match s.split_once(':') {
        Some((n, m)) => (n.trim(), Some(m.trim().to_string())),
        None => (s, None),
    };
    match name {
        "embed" | "llm" | "nli" | "jev" => Ok(Leg {
            name: name.to_string(),
            model,
        }),
        other => anyhow::bail!("unknown scorer '{other}' in route spec; want embed|llm|nli|jev"),
    }
}

/// Parse `choice=<leg>,score=<leg>,noul=<leg>`.
pub fn parse_route_spec(s: &str) -> Result<RouteSpec> {
    let mut choice = None;
    let mut score = None;
    let mut noul = None;
    for part in s.split(',') {
        let (key, leg) = part
            .split_once('=')
            .with_context(|| format!("route leg '{part}' must look like choice=<scorer[:model]>"))?;
        let leg = parse_leg(leg)?;
        match key.trim() {
            "choice" => choice = Some(leg),
            "score" => score = Some(leg),
            "noul" => noul = Some(leg),
            other => anyhow::bail!("unknown route key '{other}'; want choice|score|noul"),
        }
    }
    Ok(RouteSpec {
        choice: choice.context("route spec missing choice=<scorer[:model]>")?,
        score: score.context("route spec missing score=<scorer[:model]>")?,
        noul: noul.context("route spec missing noul=<scorer[:model]>")?,
    })
}

/// Any single scorer (concrete dispatch: the [`Scorer`] trait is not dyn-safe).
pub enum AnyScorer {
    Embed(crate::embed::EmbedScorer),
    Llm(crate::llm::LlmJudge),
    Nli(crate::nli::NliScorer),
    Jev(crate::jev::JevScorer),
}

impl Scorer for AnyScorer {
    async fn answer(&self, state: &str, question: &Question) -> Result<Answer> {
        match self {
            Self::Embed(s) => s.answer(state, question).await,
            Self::Llm(s) => s.answer(state, question).await,
            Self::Nli(s) => s.answer(state, question).await,
            Self::Jev(s) => s.answer(state, question).await,
        }
    }
}

/// Routes each question variant to its own scorer.
pub struct Router<C, S, N> {
    pub choice: C,
    pub score: S,
    pub noul: N,
}

impl<C: Scorer, S: Scorer, N: Scorer> Scorer for Router<C, S, N> {
    async fn answer(&self, state: &str, question: &Question) -> Result<Answer> {
        match question {
            Question::Choice { .. } => self.choice.answer(state, question).await,
            Question::Score { .. } => self.score.answer(state, question).await,
            Question::Noul { .. } => self.noul.answer(state, question).await,
        }
    }
}

/// A router over runtime-chosen scorers (what `--route` builds).
pub type DynRouter = Router<AnyScorer, AnyScorer, AnyScorer>;

/// Backend defaults shared by CLI and server construction.
pub struct Backends {
    pub ollama: crate::ollama::OllamaClient,
    pub embed_model: String,
    pub llm_model: String,
    pub nli_model: String,
    pub jev_api_key: Option<String>,
    pub jev_model: String,
    pub temperature: f64,
}

impl Backends {
    fn ollama(&self) -> crate::ollama::OllamaClient {
        self.ollama.clone()
    }

    /// Build one leg; the leg model overrides the default when present.
    pub fn build(&self, leg: &Leg) -> Result<AnyScorer> {
        let model = |default: &str| leg.model.clone().unwrap_or_else(|| default.to_string());
        match leg.name.as_str() {
            "embed" => {
                let mut s = crate::embed::EmbedScorer::new(self.ollama(), model(&self.embed_model));
                s.temperature = self.temperature;
                Ok(AnyScorer::Embed(s))
            }
            "llm" => Ok(AnyScorer::Llm(crate::llm::LlmJudge::new(
                self.ollama(),
                model(&self.llm_model),
            ))),
            "nli" => Ok(AnyScorer::Nli(crate::nli::NliScorer::load(&model(
                &self.nli_model,
            ))?)),
            "jev" => {
                let key = match &self.jev_api_key {
                    Some(k) => k.clone(),
                    None => crate::jev::read_api_key()?,
                };
                Ok(AnyScorer::Jev(crate::jev::JevScorer::new(
                    key,
                    model(&self.jev_model),
                )))
            }
            other => anyhow::bail!("unknown scorer '{other}'"),
        }
    }

    pub fn build_router(&self, spec: &RouteSpec) -> Result<DynRouter> {
        Ok(DynRouter {
            choice: self.build(&spec.choice)?,
            score: self.build(&spec.score)?,
            noul: self.build(&spec.noul)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn parses_full_spec() {
        let spec = parse_route_spec("choice=nli:./v7,score=jev,noul=llm:phi4").unwrap();
        assert_eq!(spec.choice.name, "nli");
        assert_eq!(spec.choice.model.as_deref(), Some("./v7"));
        assert_eq!(spec.score.name, "jev");
        assert_eq!(spec.score.model, None);
        assert_eq!(spec.noul.model.as_deref(), Some("phi4"));
    }

    #[test]
    fn rejects_bad_specs() {
        for bad in [
            "",
            "choice=nli,score=jev",                          // missing noul
            "choice=nli,score=jev,noul=xyz",                 // unknown scorer
            "choice=nli,score=jev,urgency=nli",              // unknown key
            "choice=nli score=jev,noul=nli",                 // missing '='
            "choice=,score=jev,noul=nli",                    // empty leg
        ] {
            assert!(parse_route_spec(bad).is_err(), "accepted: {bad}");
        }
    }

    struct Tag(&'static str);

    impl Scorer for Tag {
        async fn answer(&self, _state: &str, _q: &Question) -> Result<Answer> {
            Ok(Answer::Noul {
                noul: match self.0 {
                    "choice" => 0.1,
                    "score" => 0.2,
                    _ => 0.3,
                },
            })
        }
    }

    fn choice_q() -> Question {
        Question::Choice {
            instructions: String::new(),
            criteria: HashMap::from([("a".to_string(), "b".to_string())]),
        }
    }

    #[tokio::test]
    async fn routes_each_variant_to_its_leg() {
        let r = Router {
            choice: Tag("choice"),
            score: Tag("score"),
            noul: Tag("noul"),
        };
        let cn = |a: Answer| match a {
            Answer::Noul { noul } => noul,
            _ => panic!("stub must return noul"),
        };
        assert_eq!(
            cn(r.answer("s", &choice_q()).await.unwrap()),
            0.1
        );
        assert_eq!(
            cn(r
                .answer(
                    "s",
                    &Question::Score {
                        instructions: String::new(),
                        criteria: vec!["x".to_string()],
                    },
                )
                .await
                .unwrap()),
            0.2
        );
        assert_eq!(
            cn(r
                .answer("s", &Question::Noul { instructions: String::new() })
                .await
                .unwrap()),
            0.3
        );
    }
}
