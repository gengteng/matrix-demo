use std::ops::{Add, Index, IndexMut, Mul};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct DynMatrics<T, const R: usize, const C: usize> {
    data: Vec<T>,
}
impl<T, const R: usize, const C: usize> Index<usize> for DynMatrics<T, R, C> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * C;
        let end = start + C;
        &self.data[start..end]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for DynMatrics<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * C;
        let end = start + C;
        &mut self.data[start..end]
    }
}

impl<T, const X: usize, const Y: usize> Default for DynMatrics<T, X, Y>
where
    T: Default + Copy,
{
    fn default() -> Self {
        DynMatrics {
            data: vec![T::default(); X * Y],
        }
    }
}

impl<T, const X: usize, const Y: usize> DynMatrics<T, X, Y> {
    pub fn dot_product<const Z: usize>(&self, matrix1: &DynMatrics<T, Y, Z>) -> DynMatrics<T, X, Z>
    where
        T: Default + Add<Output = T> + Mul<Output = T> + Copy,
    {
        let mut result = DynMatrics::<T, X, Z>::default();
        for i in 0..X {
            for j in 0..Z {
                let mut sum = T::default();
                for k in 0..Y {
                    sum = sum + self.data[i * Y + k] * matrix1.data[k * Z + j];
                }
                result.data[i * Z + j] = sum;
            }
        }
        result
    }

    pub fn dot_product_in_parallel<const Z: usize>(
        &self,
        matrix1: &DynMatrics<T, Y, Z>,
        parallel: usize,
    ) -> DynMatrics<T, X, Z>
    where
        T: Default + Add<Output = T> + Mul<Output = T> + Copy + Send + Sync,
    {
        let mut result = DynMatrics::<T, X, Z>::default();
        let matrix0 = &self.data;
        let matrix1_data = &matrix1.data;

        std::thread::scope(|scope| {
            let chunk_size = (X + parallel - 1) / parallel; // 计算每个线程应处理的行数

            let mut start_index = 0;
            let mut result_slices = &mut result.data[..];

            for _ in 0..parallel {
                if start_index >= X {
                    break;
                }
                let chunk_end = std::cmp::min(start_index + chunk_size, X);
                let (local_result, rest) =
                    result_slices.split_at_mut((chunk_end - start_index) * Z);
                result_slices = rest;

                let local_matrix0_start = start_index * Y;
                let local_matrix0_end = chunk_end * Y;
                let local_matrix0 = &matrix0[local_matrix0_start..local_matrix0_end];

                scope.spawn(move || {
                    for i in 0..(chunk_end - start_index) {
                        for j in 0..Z {
                            let mut sum = T::default();
                            for k in 0..Y {
                                sum = sum + local_matrix0[i * Y + k] * matrix1_data[k * Z + j];
                            }
                            local_result[i * Z + j] = sum;
                        }
                    }
                });

                start_index = chunk_end;
            }
        });

        result
    }
}

impl<T, const X: usize, const Y: usize> TryFrom<Vec<T>> for DynMatrics<T, X, Y> {
    type Error = ();

    fn try_from(data: Vec<T>) -> Result<Self, ()> {
        if data.len() == X * Y {
            Ok(Self { data })
        } else {
            return Err(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_2x2() {
        let a = DynMatrics::<_, 2, 2>::try_from(vec![1, 2, 3, 4]).unwrap();
        let b = DynMatrics::<_, 2, 2>::try_from(vec![2, 0, 1, 2]).unwrap();
        let result = a.dot_product(&b);
        let expected = DynMatrics::<_, 2, 2>::try_from(vec![4, 4, 10, 8]).unwrap();
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_2x3_and_3x2() {
        let a = DynMatrics::<_, 2, 3>::try_from(vec![1, 2, 3, 4, 5, 6]).unwrap();
        let b = DynMatrics::<_, 3, 2>::try_from(vec![7, 8, 9, 10, 11, 12]).unwrap();
        let result = a.dot_product(&b);
        let expected = DynMatrics::<_, 2, 2>::try_from(vec![58, 64, 139, 154]).unwrap();
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_2x2_par() {
        let cpus = num_cpus::get();
        let a = DynMatrics::<_, 2, 2>::try_from(vec![1, 2, 3, 4]).unwrap();
        let b = DynMatrics::<_, 2, 2>::try_from(vec![2, 0, 1, 2]).unwrap();
        let result = a.dot_product_in_parallel(&b, cpus);
        let expected = DynMatrics::<_, 2, 2>::try_from(vec![4, 4, 10, 8]).unwrap();
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_product_2x3_and_3x2_par() {
        let cpus = num_cpus::get();
        let a = DynMatrics::<_, 2, 3>::try_from(vec![1, 2, 3, 4, 5, 6]).unwrap();
        let b = DynMatrics::<_, 3, 2>::try_from(vec![7, 8, 9, 10, 11, 12]).unwrap();
        let result = a.dot_product_in_parallel(&b, cpus);
        let expected = DynMatrics::<_, 2, 2>::try_from(vec![58, 64, 139, 154]).unwrap();
        assert_eq!(result.data, expected.data);
    }
}
