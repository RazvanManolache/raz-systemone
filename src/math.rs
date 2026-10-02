//! Small probability helpers shared by all scorers.

/// Cosine similarity in [-1, 1]. Returns 0 for zero/empty/mismatched vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Softmax over logits with temperature `t` (> 0). Lower `t` sharpens.
pub fn softmax(logits: &[f64], t: f64) -> Vec<f64> {
    assert!(t > 0.0, "temperature must be positive");
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| ((x - max) / t).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|x| x / sum).collect()
}

/// Logistic function.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Index of the maximum value (first wins ties).
pub fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in xs.iter().enumerate() {
        if *x > xs[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn softmax_sums_to_one_and_sharpens() {
        let p = softmax(&[1.0, 2.0, 3.0], 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(p[2] > p[1] && p[1] > p[0]);
        let sharp = softmax(&[1.0, 2.0, 3.0], 0.1);
        assert!(sharp[2] > 0.99);
    }

    #[test]
    fn sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(10.0) > 0.9999);
        assert!(sigmoid(-10.0) < 0.0001);
    }
}
