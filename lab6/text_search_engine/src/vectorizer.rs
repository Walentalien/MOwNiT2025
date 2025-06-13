// src/vectorizer.rs
use ndarray::{Array1, Array2};
use ndarray::Zip;

pub fn apply_idf(matrix: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let (m, n) = matrix.dim();
    let mut idf = Array1::<f64>::zeros(m);

    for i in 0..m {
        let df = matrix.row(i).iter().filter(|&&x| x > 0.0).count();
        idf[i] = ((n as f64) / (df as f64 + 1.0)).ln(); // smoothed
    }

    let mut tfidf = matrix.clone();
    for ((mut row), &idf_val) in tfidf.outer_iter_mut().zip(idf.iter()) {
        Zip::from(&mut row).for_each(|x| *x *= idf_val);
    }

    (tfidf, idf)
}
