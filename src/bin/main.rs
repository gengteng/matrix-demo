use matrix::dynamic::DynMatrics;
use matrix::Matrix;
use std::time::Instant;

fn main() {
    let cpus = num_cpus::get();
    println!("cpus: {}", cpus);

    let a = generate_matrix::<16, 10000>();
    let b = generate_matrix::<10000, 1000>();

    let start = Instant::now();
    let _result_sequential = a.dot_product(&b);
    let duration_sequential = start.elapsed();

    let start = Instant::now();
    let _result_parallel = a.dot_product_in_parallel(&b, cpus);
    let duration_parallel = start.elapsed();

    println!("Sequential dot product took: {:?}", duration_sequential);
    println!("Parallel dot product took: {:?}", duration_parallel);

    let a = generate_dynamic_matrix::<16, 10000>();
    let b = generate_dynamic_matrix::<10000, 1000>();

    let start = Instant::now();
    let _result_sequential = a.dot_product(&b);
    let duration_sequential = start.elapsed();

    let start = Instant::now();
    let _result_parallel = a.dot_product_in_parallel(&b, cpus);
    let duration_parallel = start.elapsed();

    println!(
        "(dyn) Sequential dot product took: {:?}",
        duration_sequential
    );
    println!("(dyn) Parallel dot product took: {:?}", duration_parallel);
}

fn generate_matrix<const X: usize, const Y: usize>() -> Matrix<i32, X, Y> {
    let mut matrix = Matrix::default();
    for i in 0..X {
        for j in 0..Y {
            matrix[i][j] = rand::random::<u8>() as _;
        }
    }
    matrix
}

fn generate_dynamic_matrix<const X: usize, const Y: usize>() -> DynMatrics<i32, X, Y> {
    let mut matrix = DynMatrics::default();
    for i in 0..X {
        for j in 0..Y {
            matrix[i][j] = rand::random::<u8>() as _;
        }
    }
    matrix
}
