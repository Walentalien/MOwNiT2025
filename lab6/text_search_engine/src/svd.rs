// src/svd.rs
use ndarray::{Array2, s};
use ndarray_linalg::SVD;

pub fn low_rank_approx(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let (u, s, vt) = matrix.svd(true, true).unwrap();
    let u = u.unwrap().slice(s![.., 0..k]).to_owned();
    let vt = vt.unwrap().slice(s![0..k, ..]).to_owned();
    let s_mat = Array2::from_diag(&s.slice(s![0..k]));

    u.dot(&s_mat).dot(&vt)
}
