pub mod dynamic;

use std::ops::{Add, Index, IndexMut, Mul};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Matrix<T, const R: usize, const C: usize> {
    data: Box<[[T; C]; R]>,
}

impl<T, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = [T; C];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const X: usize, const Y: usize> Default for Matrix<T, X, Y>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Matrix {
            data: Box::new([[T::default(); Y]; X]),
        }
    }
}

impl<T, const X: usize, const Y: usize> Matrix<T, X, Y> {
    pub fn dot_product<const Z: usize>(&self, matrix1: &Matrix<T, Y, Z>) -> Matrix<T, X, Z>
    where
        T: Default + Add<Output = T> + Mul<Output = T> + Copy,
    {
        let mut result = Matrix::<T, X, Z>::default();
        for i in 0..X {
            for j in 0..Z {
                let mut sum = T::default();
                for k in 0..Y {
                    sum = sum + self.data[i][k] * matrix1.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn dot_product_in_parallel<const Z: usize>(
        &self,
        matrix1: &Matrix<T, Y, Z>,
        parallel: usize,
    ) -> Matrix<T, X, Z>
    where
        T: Default + Add<Output = T> + Mul<Output = T> + Copy + Send + Sync,
    {
        let mut result = Matrix::<T, X, Z>::default();
        let matrix0 = &self.data;
        let matrix1_data = &matrix1.data;

        std::thread::scope(|scope| {
            let chunk_size = (X + parallel - 1) / parallel; // 计算每个线程应处理的行数
            result
                .data
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(i, chunk)| {
                    scope.spawn(move || {
                        let start_index = i * chunk_size; // 计算全局行的起始索引
                        for (local_index, row) in chunk.iter_mut().enumerate() {
                            let global_index = start_index + local_index; // 计算全局行索引
                            for z in 0..Z {
                                let mut sum = T::default();
                                for y in 0..Y {
                                    sum = sum + matrix0[global_index][y] * matrix1_data[y][z];
                                }
                                row[z] = sum;
                            }
                        }
                    });
                });
        });

        result
    }
}

impl<T, const X: usize, const Y: usize> From<[[T; Y]; X]> for Matrix<T, X, Y> {
    fn from(data: [[T; Y]; X]) -> Self {
        Self {
            data: Box::new(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_2x2() {
        let a = Matrix::from([[1, 2], [3, 4]]);
        let b = Matrix::from([[2, 0], [1, 2]]);
        let result = a.dot_product(&b);
        let expected = Matrix::from([[4, 4], [10, 8]]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_2x3_and_3x2() {
        let a = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let b = Matrix::from([[7, 8], [9, 10], [11, 12]]);
        let result = a.dot_product(&b);
        let expected = Matrix::from([[58, 64], [139, 154]]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_identity_matrix() {
        let a = Matrix::from([[1, 0], [0, 1]]);
        let b = Matrix::from([[5, 6], [7, 8]]);
        let result = a.dot_product(&b);
        assert_eq!(result.data, b.data);
    }

    #[test]
    fn test_dot_product_with_zero_matrix() {
        let a = Matrix::from([[0, 0], [0, 0]]);
        let b = Matrix::from([[1, 2], [3, 4]]);
        let result = a.dot_product(&b);
        let expected = Matrix::from([[0, 0], [0, 0]]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_2x2_par() {
        let a = Matrix::from([[1, 2], [3, 4]]);
        let b = Matrix::from([[2, 0], [1, 2]]);
        let result = a.dot_product_in_parallel(&b, num_cpus::get());
        let expected = Matrix::from([[4, 4], [10, 8]]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_2x3_and_3x2_par() {
        let a = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let b = Matrix::from([[7, 8], [9, 10], [11, 12]]);
        let result = a.dot_product_in_parallel(&b, num_cpus::get());
        let expected = Matrix::from([[58, 64], [139, 154]]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_identity_matrix_par() {
        let a = Matrix::from([[1, 0], [0, 1]]);
        let b = Matrix::from([[5, 6], [7, 8]]);
        let result = a.dot_product_in_parallel(&b, num_cpus::get());
        assert_eq!(result.data, b.data);
    }

    #[test]
    fn test_dot_product_with_zero_matrix_par() {
        let a = Matrix::from([[0, 0], [0, 0]]);
        let b = Matrix::from([[1, 2], [3, 4]]);
        let result = a.dot_product_in_parallel(&b, num_cpus::get());
        let expected = Matrix::from([[0, 0], [0, 0]]);
        assert_eq!(result.data, expected.data);
    }
}
