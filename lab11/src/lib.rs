// Compute the dot product of two vectors of the same length
pub fn dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v.iter()).map(|(ui, vi)| ui * vi).sum()
}

// Compute the euclidean norm of a vector
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
// 1.1 Subtask completed
pub fn gram_schmidt(a: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    assert!(a.iter().all(|row| row.len() == n), "Matrix must be square");

    // Initialize Q and R with zeros
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];

    // Work column by column
    for j in 0..n {
        // Copy j-th column of A into v
        let mut v: Vec<f64> = (0..n).map(|i| a[i][j]).collect();

        // Subtract projections onto previous q-columns
        for i in 0..j {
            // r[i][j] = q[:,i]·a[:,j]
            let qi: Vec<f64> = (0..n).map(|k| q[k][i]).collect();
            r[i][j] = dot(&qi, &v);
            // v = v - r[i][j] * q[:,i]
            for k in 0..n {
                v[k] -= r[i][j] * q[k][i];
            }
        }

        // r[j][j] = ||v||
        r[j][j] = norm(&v);

        // Normalize to get the j-th column of Q
        for k in 0..n {
            q[k][j] = v[k] / r[j][j];
        }
    }
    (q, r)
}
