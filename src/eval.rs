//! Labeled-set evaluation: accuracy plus expected calibration error (ECE).
//!
//! Each case pins expected answers for the standard ticket questions; any
//! field may be `None` when the state genuinely doesn't cover it.

use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::{Answer, AskRequest, Question, Scorer};

/// One labeled case. Labels were fixed by human judgment before any model ran.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Case {
    pub id: String,
    pub state: String,
    #[serde(default)]
    pub department: Option<String>,
    #[serde(default)]
    pub frustration: Option<u8>,
    #[serde(default)]
    pub is_urgent: Option<bool>,
}

/// Parse JSONL (one case per line, blank lines skipped).
pub fn load_cases(text: &str) -> Result<Vec<Case>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).with_context(|| format!("line {}", i + 1)))
        .collect()
}

/// The standard questions every case is judged on.
fn questions() -> HashMap<String, Question> {
    let mut q = HashMap::new();
    q.insert(
        "department".to_string(),
        Question::Choice {
            instructions: "Which team should handle this".to_string(),
            criteria: [
                ("billing", "Payment or subscription issues"),
                ("technical", "Bugs or integration problems"),
                ("sales", "Pricing or account questions"),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        },
    );
    q.insert(
        "frustration".to_string(),
        Question::Score {
            instructions: "How frustrated the customer appears".to_string(),
            criteria: vec![
                "Calm, just stating facts".to_string(),
                "Frustrated but civil".to_string(),
                "Very angry, strong language".to_string(),
            ],
        },
    );
    q.insert(
        "is_urgent".to_string(),
        Question::Noul {
            instructions: "The message conveys urgency or time-sensitivity".to_string(),
        },
    );
    q
}

#[derive(Debug, Clone)]
struct Judgment {
    id: String,
    question: &'static str,
    confidence: f64,
    correct: bool,
    detail: String,
    answer: Answer,
    expected: serde_json::Value,
}

/// One JSON-serializable dump line per judgment (for offline fitting).
fn dump_line(j: &Judgment) -> serde_json::Value {
    serde_json::json!({
        "id": j.id,
        "question": j.question,
        "correct": j.correct,
        "answer": j.answer,
        "expected": j.expected,
    })
}

pub struct Report {
    judgments: Vec<Judgment>,
}

impl Report {
    pub fn accuracy(&self) -> f64 {
        mean(&self.judgments.iter().map(|j| j.correct as u8 as f64).collect::<Vec<_>>())
    }

    pub fn accuracy_for(&self, question: &str) -> Option<f64> {
        let xs: Vec<f64> = self
            .judgments
            .iter()
            .filter(|j| j.question == question)
            .map(|j| j.correct as u8 as f64)
            .collect();
        if xs.is_empty() {
            None
        } else {
            Some(mean(&xs))
        }
    }

    /// Expected calibration error over 10 equal-width confidence bins.
    pub fn ece(&self) -> f64 {
        ece(
            &self.judgments.iter().map(|j| (j.confidence, j.correct)).collect::<Vec<_>>(),
            10,
        )
    }

    /// Write one JSON line per judgment for offline analysis/fitting.
    pub fn write_dump(&self, path: &std::path::Path) -> Result<()> {
        use std::fmt::Write as _;
        let mut out = String::new();
        for j in &self.judgments {
            writeln!(out, "{}", dump_line(j)).unwrap();
        }
        std::fs::write(path, out)?;
        Ok(())
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.judgments.len();
        let misses: Vec<&Judgment> = self.judgments.iter().filter(|j| !j.correct).collect();
        writeln!(f, "judgments: {n}")?;
        writeln!(
            f,
            "accuracy: {:.3} overall | choice {:.3} | score {:.3} | noul {:.3}",
            self.accuracy(),
            self.accuracy_for("department").unwrap_or(f64::NAN),
            self.accuracy_for("frustration").unwrap_or(f64::NAN),
            self.accuracy_for("is_urgent").unwrap_or(f64::NAN),
        )?;
        writeln!(f, "ECE: {:.3}", self.ece())?;
        writeln!(f, "reliability (conf range, n, acc):")?;
        for b in 0..10 {
            let lo = b as f64 / 10.0;
            let in_bin: Vec<&Judgment> = self
                .judgments
                .iter()
                .filter(|j| j.confidence >= lo && (j.confidence < lo + 0.1 || lo >= 0.9))
                .collect();
            if in_bin.is_empty() {
                continue;
            }
            let acc = in_bin.iter().filter(|j| j.correct).count() as f64 / in_bin.len() as f64;
            writeln!(f, "  {lo:.1}-{:.1}  {:>3}  {acc:.2}", lo + 0.1, in_bin.len())?;
        }
        writeln!(f, "misses ({}):", misses.len())?;
        for m in &misses {
            writeln!(f, "  {}", m.detail)?;
        }
        Ok(())
    }
}

pub async fn run<S: Scorer>(scorer: &S, cases: &[Case]) -> Result<Report> {
    let mut judgments = Vec::new();
    for case in cases {
        let resp = crate::evaluate(
            scorer,
            &AskRequest {
                state: case.state.clone(),
                questions: questions(),
            },
        )
        .await
        .with_context(|| format!("case {}", case.id))?;
        if let Some(exp) = &case.department {
            match resp.answers.get("department") {
                Some(a @ Answer::Choice {
                    choice,
                    confidence,
                    ..
                }) => judgments.push(Judgment {
                    id: case.id.clone(),
                    question: "department",
                    confidence: *confidence,
                    correct: choice == exp,
                    detail: format!(
                        "[{}] department: got {choice} ({confidence:.2}), want {exp}",
                        case.id
                    ),
                    answer: a.clone(),
                    expected: serde_json::Value::String(exp.clone()),
                }),
                other => anyhow::bail!("case {}: bad department answer: {other:?}", case.id),
            }
        }
        if let Some(exp) = case.frustration {
            match resp.answers.get("frustration") {
                Some(a @ Answer::Score {
                    score, confidence, ..
                }) => judgments.push(Judgment {
                    id: case.id.clone(),
                    question: "frustration",
                    confidence: *confidence,
                    // Round: Jev returns an expected level (e.g. 0.98 ≈ 1).
                    correct: score.round() as u8 == exp,
                    detail: format!(
                        "[{}] frustration: got {score} ({confidence:.2}), want {exp}",
                        case.id
                    ),
                    answer: a.clone(),
                    expected: serde_json::Value::from(exp),
                }),
                other => anyhow::bail!("case {}: bad frustration answer: {other:?}", case.id),
            }
        }
        if let Some(exp) = case.is_urgent {
            match resp.answers.get("is_urgent") {
                Some(a @ Answer::Noul { noul }) => {
                    let pred = *noul >= 0.5;
                    judgments.push(Judgment {
                        id: case.id.clone(),
                        question: "is_urgent",
                        confidence: noul.max(1.0 - noul),
                        correct: pred == exp,
                        detail: format!(
                            "[{}] is_urgent: got {noul:.2} (-> {pred}), want {exp}",
                            case.id
                        ),
                        answer: a.clone(),
                        expected: serde_json::Value::Bool(exp),
                    })
                }
                other => anyhow::bail!("case {}: bad is_urgent answer: {other:?}", case.id),
            }
        }
    }
    Ok(Report { judgments })
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn ece(items: &[(f64, bool)], bins: usize) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let mut err = 0.0;
    for b in 0..bins {
        let lo = b as f64 / bins as f64;
        let in_bin: Vec<&(f64, bool)> = items
            .iter()
            .filter(|(c, _)| *c >= lo && (*c < lo + 1.0 / bins as f64 || lo >= (bins - 1) as f64 / bins as f64))
            .collect();
        if in_bin.is_empty() {
            continue;
        }
        let acc = in_bin.iter().filter(|(_, ok)| *ok).count() as f64 / in_bin.len() as f64;
        let conf = in_bin.iter().map(|(c, _)| c).sum::<f64>() / in_bin.len() as f64;
        err += (acc - conf).abs() * in_bin.len() as f64 / items.len() as f64;
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ece_zero_when_perfect() {
        let items = vec![(0.95, true), (0.95, true), (0.05, false), (0.05, false)];
        assert!(ece(&items, 10) < 0.06);
    }

    #[test]
    fn ece_high_when_overconfident() {
        let items = vec![(0.95, false); 10];
        assert!(ece(&items, 10) > 0.9);
    }

    #[test]
    fn dump_line_shape() {
        let j = Judgment {
            id: "t1".to_string(),
            question: "is_urgent",
            confidence: 0.9,
            correct: true,
            detail: String::new(),
            answer: Answer::Noul { noul: 0.9 },
            expected: serde_json::Value::Bool(true),
        };
        let v = dump_line(&j);
        assert_eq!(v["id"], "t1");
        assert_eq!(v["answer"]["noul"], 0.9);
        assert_eq!(v["expected"], true);
    }

    #[test]
    fn load_skips_blanks() {
        let cases = load_cases("{\"id\":\"a\",\"state\":\"x\"}\n\n{\"id\":\"b\",\"state\":\"y\"}\n").unwrap();
        assert_eq!(cases.len(), 2);
        assert!(cases[0].department.is_none());
    }
}
