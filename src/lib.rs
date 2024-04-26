use std::ops::{Add, Mul};
use std::slice::SplitMut;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct Matrix<T, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T, const X: usize, const Y: usize> Default for Matrix<T, X, Y>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Matrix {
            data: [[T::default(); Y]; X],
        }
    }
}

impl<T, const X: usize, const Y: usize> Matrix<T, X, Y> {
    pub fn split_mut<F>(&mut self, pred: F) -> SplitMut<'_, [T; Y], F>
    where
        F: FnMut(&[T; Y]) -> bool,
    {
        self.data.split_mut(pred)
    }

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
    ) -> Matrix<T, X, Z>
    where
        T: Default + Add<Output = T> + Mul<Output = T> + Copy + Send + Sync,
    {
        let mut result = Matrix::<T, X, Z>::default();

        std::thread::scope(|scope| {
            for (x, row) in result.data.iter_mut().enumerate() {
                let matrix0_row = &self.data[x];
                scope.spawn(move || {
                    for (z, item1) in row.iter_mut().enumerate() {
                        let mut sum = T::default();
                        for (y, item0) in matrix0_row.iter().enumerate() {
                            sum = sum + (*item0 * matrix1.data[y][z]);
                        }
                        *item1 = sum;
                    }
                });
            }
        });

        result
    }
}

impl<T, const X: usize, const Y: usize> From<[[T; Y]; X]> for Matrix<T, X, Y> {
    fn from(data: [[T; Y]; X]) -> Self {
        Self { data }
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
        let result = a.dot_product_in_parallel(&b);
        let expected = Matrix::from([[4, 4], [10, 8]]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_2x3_and_3x2_par() {
        let a = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let b = Matrix::from([[7, 8], [9, 10], [11, 12]]);
        let result = a.dot_product_in_parallel(&b);
        let expected = Matrix::from([[58, 64], [139, 154]]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_identity_matrix_par() {
        let a = Matrix::from([[1, 0], [0, 1]]);
        let b = Matrix::from([[5, 6], [7, 8]]);
        let result = a.dot_product_in_parallel(&b);
        assert_eq!(result.data, b.data);
    }

    #[test]
    fn test_dot_product_with_zero_matrix_par() {
        let a = Matrix::from([[0, 0], [0, 0]]);
        let b = Matrix::from([[1, 2], [3, 4]]);
        let result = a.dot_product_in_parallel(&b);
        let expected = Matrix::from([[0, 0], [0, 0]]);
        assert_eq!(result.data, expected.data);
    }
}
