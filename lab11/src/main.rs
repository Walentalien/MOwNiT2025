use nalgebra::{DMatrix, DVector, SVD};
use nalgebra::linalg::QR;
use rand::{thread_rng, Rng};
use rand::distr::Uniform;
use lab11::*;
use plotters::prelude::*;
//code for task 2
fn qr_decompose(a: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    let n = a[0].len();
    let mut q = vec![vec![0.0; n]; m];
    let mut r = vec![vec![0.0; n]; n];

    for j in 0..n {
        //for column j
        let mut v = vec![0.0; m];
        for i in 0..m {
            v[i] = a[i][j];
        }
        // projection on previous vectors
        for i in 0..j {
            let mut dot = 0.0;
            for k in 0..m {
                dot += q[k][i] * v[k];
            }
            r[i][j] = dot;
            for k in 0..m {
                v[k] -= dot * q[k][i];
            }
        }
        // nirm and normalizationS
        let mut norm = 0.0;
        for k in 0..m {
            norm += v[k] * v[k];
        }
        let norm = norm.sqrt();
        r[j][j] = norm;
        if norm > 1e-12 {
            for k in 0..m {
                q[k][j] = v[k] / norm;
            }
        }
    }
    (q, r)
}

fn solve_least_squares_qr(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let (q, r) = qr_decompose(a);
    let m = a.len();
    let n = a[0].len();

    // calculate Q^T b
    let mut qt_b = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += q[k][i] * b[k];
        }
        qt_b[i] = sum;
    }

    // solve R x = Q^T b with back substitution
    let mut x = vec![0.0; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        if r[i][i].abs() < 1e-12 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += r[i][j] * x[j];
        }
        x[i] = (qt_b[i] - sum) / r[i][i];
    }
    x
}

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
    println!("Qtrans * Q =\n{}", q.transpose() * q);

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

    // ======= 1.2 Subtask ========
    let mut rng = rand::thread_rng();
    let sizes = [5, 10, 50, 100];

    for &n in &sizes {
        // generate A as Vec<Vec<f64>> and as a DMatrix
        let mut a_data = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                //a_data[i][j] = rng.sample(Uniform::new_inclusive(-10.0, 10.0));
                a_data[i][j] = rng.gen_range(-10.0..10.0);
            }
        }
        let flat: Vec<f64> = a_data.iter().flat_map(|r| r.clone()).collect();
        let a = DMatrix::from_row_slice(n, n, &flat);

        // our GS
        let (q_self, r_self) = gram_schmidt(&a_data);
        let q_flat: Vec<f64> = (0..n).flat_map(|i| q_self[i].clone()).collect();
        let r_flat: Vec<f64> = (0..n).flat_map(|i| r_self[i].clone()).collect();
        let q_self_mat = DMatrix::from_row_slice(n, n, &q_flat);
        let r_self_mat = DMatrix::from_row_slice(n, n, &r_flat);

        // library QR
        let qr = QR::new(a.clone());
        let q_lib = qr.q();
        let r_lib = qr.r();

        // residuals
        let err_self = (q_self_mat.clone() * r_self_mat.clone() - a.clone()).norm();
        let err_lib  = (q_lib.clone()      * r_lib.clone()      - a.clone()).norm();
        // orthogonality
        let ortho_self = (q_self_mat.transpose() * &q_self_mat
                          - DMatrix::identity(n, n)).norm();
        let ortho_lib  = (q_lib.transpose()      * &q_lib
                          - DMatrix::identity(n, n)).norm();

        println!("n = {}", n);
        /*
        Considering the size of error -> the function is working
        (errors are clse to the machine epsilon (2.2*10^-16)
        This error come from round-ups in double presizion aritmetic
        Library function has better results expecially for bigger n
         */
        println!("  GS:  ‖QR−A‖ = {:8.3e},  ‖QTrans Q−I‖ = {:8.3e}", err_self, ortho_self);
        println!("  lib: ‖QR−A‖ = {:8.3e},  ‖Qtrans Q−I‖ = {:8.3e}", err_lib,  ortho_lib);
        println!();
    }
    // ======== 1.2 End of subtask========
    //======== 1.3 Subtask starts here ========
    fn generate_matrices(
        n: usize,
        num: usize,
        cond_min: f64,
        cond_max: f64,
    ) -> Vec<DMatrix<f64>> {
        let mut rng = thread_rng();
        let log_min = cond_min.log10();
        let log_max = cond_max.log10();

        (0..num).map(|i| {
            // pick condition number on log‐scale
            let t = i as f64 / (num as f64 - 1.0);
            let cond = 10f64.powf(log_min + t * (log_max - log_min));

            // build diagonal of singular values: [cond, …, 1]
            let sigmas: Vec<f64> = (0..n)
                .map(|k| {
                    let exponent = (n - 1 - k) as f64 / (n as f64 - 1.0);
                    cond.powf(exponent)
                })
                .collect();
            let D = DMatrix::from_diagonal(&DVector::from_vec(sigmas));

            // random Gaussian matrix -> SVD -> U
            let M1 = DMatrix::from_fn(n, n, |_, _| rng.gen_range(-1.0..1.0));
            let svd1 = SVD::new(M1, true, false);
            let U = svd1.u.expect("SVD failed to produce U");

            // random Gaussian matrix -> QR -> Q = V
            let M2 = DMatrix::from_fn(n, n, |_, _| rng.gen_range(-1.0..1.0));
            let qr2 = QR::new(M2);
            let V = qr2.q();

            // assemble A = U * D * Vᵀ
            U * D * V.transpose()
        })
            .collect()
    }

    let n = 8;
    let num = 40;
    let cond_min = 1e1;
    let cond_max = 1e10;

    let matrices = generate_matrices(n, num, cond_min, cond_max);

    for (i, A) in matrices.iter().enumerate() {
        // Optionally compute the actual condition number via SVD:
        let svd = SVD::new(A.clone(), false, false);
        let s = &svd.singular_values;
        let cond_est = s[0] / s[n - 1];

        println!(
            "Matrix {:2}: approx cond(A) = {:8.2e}",
            i + 1,
            cond_est
        );

    }

    let matrices = generate_matrices(n, num, cond_min, cond_max);

    println!(" idx   cond(A)        ‖I - Qtrans *Q‖_F");
    println!("---------------------------------");
    // ======== 1.3 End  of subtask========
    // 2) for each A_i, compute cond and GS-orthogonality error
    for (i, a) in matrices.iter().enumerate() {
        // compute actual singular values to estimate cond(A)
        let svd = SVD::new(a.clone(), false, false);
        let s = &svd.singular_values;
        let cond_est = s[0] / s[n - 1];

        // convert A (DMatrix) into Vec<Vec<f64>>
        let mut a_data = vec![vec![0.0; n]; n];
        for row in 0..n {
            for col in 0..n {
                a_data[row][col] = a[(row, col)];
            }
        }

        // run classical Gram–Schmidt
        let (q_self, _) = gram_schmidt(&a_data);

        // build Q_self as DMatrix from row-major q_self
        let flat_q: Vec<f64> = q_self.iter().flat_map(|r| r.clone()).collect();
        let q_mat = DMatrix::from_row_slice(n, n, &flat_q);

        // orthonormality error: ‖I - QᵀQ‖_F
        let ident = DMatrix::identity(n, n);
        let ortho_err = (ident - q_mat.transpose() * &q_mat).norm();

        println!("{:3}   {:9.2e}   {:9.2e}", i + 1, cond_est, ortho_err);
    }
    // ======== 1.4 End  of subtask==
    /*
    Fazit:
        Error grows with cond(A)\
        Most of error comes form "projection+substraction step of algprithm"

     */
    //  // ======== 1.5 End  of subtask==

// Task 2

    let xs = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = vec![ 2.0,  7.0,  9.0, 12.0, 13.0,14.0,14.0,13.0,10.0, 8.0, 4.0];

    let m = xs.len();
    // build a as matrix m×3: [1, x, x^2]
    let mut a = vec![vec![0.0; 3]; m];
    for i in 0..m {
        a[i][0] = 1.0;
        a[i][1] = xs[i];
        a[i][2] = xs[i] * xs[i];
    }

    let alpha = solve_least_squares_qr(&a, &ys);

    println!("Elements of model f(x) = α0 + α1 x + α2 x^2:");
    println!("  α0 = {:.8}", alpha[0]);
    println!("  α1 = {:.8}", alpha[1]);
    println!("  α2 = {:.8}", alpha[2]);

    println!("Creating visualization");
    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).expect("TODO: panic message");
    let mut chart = ChartBuilder::on(&root)
        .caption("approx middlesq", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-5.5..5.5, 0.0..15.0).expect("");

    chart.configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw().expect("drawing plot failed");

    chart.draw_series(
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| Circle::new((x, y), 5, BLUE.filled()))
    ).expect("");
}
