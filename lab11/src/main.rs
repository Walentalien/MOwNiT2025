use nalgebra::{DMatrix};
use nalgebra::linalg::QR;
use lab11::*;
fn main() {
    let a = DMatrix::<f64>::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ]);
    println!("Performing QR Factorization using nalgebra crait");
    let qr = QR::new(a.clone());
    let q = qr.q();  // the orthogonal factor Q
    let r = qr.r();  // the upper triangular factor R

    println!("A =\n{}", a);
    println!("Q =\n{}", q);
    println!("R =\n{}", r);

    // quick check: Q^t * Q almost I ()
    println!("QᵀQ =\n{}", q.transpose() * q);

// testing random matrx turned out to be rank-defficient!!!
// So my version is not quarding against near-zero norms
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
        //it works with this one
    let a = vec![
        vec![1.0, 2.0, 4.0],
        vec![4.0, 3.0, 6.0],
        vec![7.0, 8.0, 6.0],
    ];

    let (q, r) = gram_schmidt(&a);

    println!("A =");
    for row in &a {
        println!("{:8.4?}", row);
    }
    println!("\nQ =");
    for row in &q {
        println!("{:8.4?}", row);
    }
    println!("\nR =");
    for row in &r {
        println!("{:8.4?}", row);
    }

    // Verify Qtrans * Q  approx is  I
    let mut qtq = vec![vec![0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            qtq[i][j] = dot(&q.iter().map(|row| row[i]).collect::<Vec<_>>(),
                           &q.iter().map(|row| row[j]).collect::<Vec<_>>());
        }
    }
    println!("\nQtrans * Q =");
    for row in &qtq {
        println!("{:8.4?}", row);
    }
}
