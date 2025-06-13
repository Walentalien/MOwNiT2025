// src/corpus.rs
use anyhow::Result;
use regex::Regex;
use std::{collections::HashMap, fs, path::Path};
use ndarray::Array2;

pub fn load_corpus(corpus_path: &Path) -> Result<(Vec<String>, Array2<f64>, Vec<String>)> {
    let mut vocab_map: HashMap<String, usize> = HashMap::new();
    let mut doc_names = Vec::new();
    let mut docs = Vec::new();
    let re = Regex::new(r"[A-Za-z]+").unwrap();

    for entry in fs::read_dir(corpus_path)? {
        let path = entry?.path();
        if path.extension().unwrap_or_default() == "txt" {
            let content = fs::read_to_string(&path)?.to_lowercase();
            let words: Vec<String> = re
                .find_iter(&content)
                .map(|m| m.as_str().to_string())
                .collect();

            doc_names.push(path.file_name().unwrap().to_string_lossy().to_string());
            docs.push(words);
        }
    }

    // Build vocabulary
    for doc in &docs {
        for word in doc {
            if !vocab_map.contains_key(word) {
                vocab_map.insert(word.clone(), vocab_map.len());
            }
        }
    }

    let vocab_size = vocab_map.len();
    let doc_count = docs.len();
    let mut matrix = Array2::<f64>::zeros((vocab_size, doc_count));

    for (j, doc) in docs.iter().enumerate() {
        let mut freq = HashMap::new();
        for word in doc {
            *freq.entry(word).or_insert(0) += 1;
        }

        for (word, count) in freq {
            if let Some(&i) = vocab_map.get(word) {
                matrix[[i, j]] = count as f64;
            }
        }
    }

    let vocab: Vec<_> = {
        let mut temp = vec!["".to_string(); vocab_map.len()];
        for (word, idx) in vocab_map {
            temp[idx] = word;
        }
        temp
    };

    Ok((doc_names, matrix, vocab))
}
