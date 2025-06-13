// src/main.rs
mod corpus;
mod vectorizer;
mod search;
mod svd;
mod crawler;

use clap::Parser;
use search::search_query;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Query string
    query: String,

    /// Path to corpus directory
    #[arg(short, long, default_value = "data")]
    corpus: PathBuf,

    /// Rank for low-rank SVD approximation (0 = no SVD)
    #[arg(short, long, default_value_t = 0)]
    rank: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Step 0: Crawl if needed
    crawler::run_crawler();

    // Step 1: Load corpus
    let (doc_names, term_doc_matrix, vocab) = corpus::load_corpus(&args.corpus)?;
    let (idf_matrix, idf_vec) = vectorizer::apply_idf(&term_doc_matrix);

    // Step 2: Run search query
    let results = search_query(
        &args.query,
        &doc_names,
        &idf_matrix,
        &vocab,
        &idf_vec,
        args.rank,
    )?;

    println!("Top matching documents:");
    for (score, name) in results {
        println!("{:.4} - {}", score, name);
    }

    Ok(())
}
