use nalgebra::{DMatrix};
use nalgebra::linalg::QR;

fn main() {
    let a = DMatrix::<f64>::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ]);
    println!("Performing QR Factorization using nalgebra crait")
    let qr = QR::new(a.clone());
    let q = qr.q();  // the orthogonal factor Q
    let r = qr.r();  // the upper triangular factor R

    println!("A =\n{}", a);
    println!("Q =\n{}", q);
    println!("R =\n{}", r);

    // quick check: Q^t * Q almost I ()
    println!("QᵀQ =\n{}", q.transpose() * q);
}
