// src/search.rs
use crate::svd::low_rank_approx;
use anyhow::Result;
use ndarray::{Array1, Array2};
use regex::Regex;
use std::collections::HashMap;

pub fn search_query(
    query: &str,
    doc_names: &[String],
    matrix: &Array2<f64>,
    vocab: &[String],
    idf: &Array1<f64>,
    k: usize,
) -> Result<Vec<(f64, String)>> {
    let mut q_vec = Array1::<f64>::zeros(vocab.len());
    let re = Regex::new(r"[A-Za-z]+")?;
    let mut freq = HashMap::new();

    for m in re.find_iter(query) {
        let word = m.as_str().to_lowercase();
        *freq.entry(word).or_insert(0) += 1;
    }

    for (word, count) in freq {
        if let Some(idx) = vocab.iter().position(|v| v == &word) {
            q_vec[idx] = count as f64 * idf[idx];
        }
    }

    normalize_vector(&mut q_vec);

    let matrix = if k > 0 {
        low_rank_approx(matrix, k)
    } else {
        matrix.clone()
    };

    let mut results = Vec::new();
    for (j, col) in matrix.axis_iter(ndarray::Axis(1)).enumerate() {
        let mut d = col.to_owned();
        normalize_vector(&mut d);
        let sim = q_vec.dot(&d);
        results.push((sim, doc_names[j].clone()));
    }

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    Ok(results.into_iter().take(10).collect())
}

fn normalize_vector(v: &mut Array1<f64>) {
    let norm = v.dot(v).sqrt();
    if norm > 0.0 {
        *v /= norm;
    }
}
