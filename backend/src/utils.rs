use std::{
    iter::Sum,
    ops::{Add, Sub},
};

use fiat_shamir::*;
use p3_field::*;
use rayon::prelude::*;

pub fn pack_extension<EF: ExtensionField<PF<EF>>>(slice: &[EF]) -> Vec<EFPacking<EF>> {
    slice
        .par_chunks_exact(packing_width::<EF>())
        .map(EFPacking::<EF>::from_ext_slice)
        .collect::<Vec<_>>()
}

pub fn unpack_extension<EF: ExtensionField<PF<EF>>>(vec: &[EFPacking<EF>]) -> Vec<EF> {
    vec.par_iter()
        .flat_map(|x| {
            let packed_coeffs = x.as_basis_coefficients_slice();
            (0..packing_width::<EF>())
                .map(|i| EF::from_basis_coefficients_fn(|j| packed_coeffs[j].as_slice()[i]))
                .collect::<Vec<_>>()
        })
        .collect()
}

pub const fn packing_log_width<EF: Field>() -> usize {
    packing_width::<EF>().ilog2() as usize
}

pub const fn packing_width<EF: Field>() -> usize {
    PFPacking::<EF>::WIDTH
}

pub fn batch_fold_multilinears<
    EF: PrimeCharacteristicRing + Copy + Send + Sync,
    IF: Copy + Sub<Output = IF> + Send + Sync,
    OF: Copy + Add<IF, Output = OF> + Send + Sync + Sum,
    F: Fn(IF, EF) -> OF + Sync + Send,
>(
    polys: &[&[IF]],
    scalars: &[EF],
    mul_if_of: F,
) -> Vec<Vec<OF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear(poly, scalars, &mul_if_of))
        .collect()
}

pub fn fold_multilinear<
    EF: PrimeCharacteristicRing + Copy + Send + Sync,
    IF: Copy + Sub<Output = IF> + Send + Sync,
    OF: Copy + Add<IF, Output = OF> + Send + Sync + Sum,
    F: Fn(IF, EF) -> OF + Sync + Send,
>(
    m: &[IF],
    scalars: &[EF],
    mul_if_of: &F,
) -> Vec<OF> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.len());
    let new_size = m.len() / scalars.len();

    if scalars.len() == 2 {
        assert_eq!(scalars[0], EF::ONE - scalars[1]);
        let alpha = scalars[1];
        return (0..new_size)
            .into_par_iter()
            .map(|i| mul_if_of(m[i + new_size] - m[i], alpha) + m[i])
            .collect();
    }

    (0..new_size)
        .into_par_iter()
        .map(|i| {
            scalars
                .iter()
                .enumerate()
                .map(|(j, s)| mul_if_of(m[i + j * new_size], *s))
                .sum()
        })
        .collect()
}

/// Returns a vector of uninitialized elements of type `A` with the specified length.
/// # Safety
/// Entries should be overwritten before use.
#[must_use]
pub unsafe fn uninitialized_vec<A>(len: usize) -> Vec<A> {
    #[allow(clippy::uninit_vec)]
    unsafe {
        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);
        vec
    }
}

pub fn parallel_clone<A: Clone + Send + Sync>(src: &[A], dst: &mut [A]) {
    if src.len() < 1 << 15 {
        // sequential copy
        dst.clone_from_slice(src);
    } else {
        assert_eq!(src.len(), dst.len());
        let chunk_size = src.len() / rayon::current_num_threads().max(1);
        dst.par_chunks_mut(chunk_size)
            .zip(src.par_chunks(chunk_size))
            .for_each(|(d, s)| {
                d.clone_from_slice(s);
            });
    }
}

#[must_use]
pub fn parallel_clone_vec<A: Clone + Send + Sync>(vec: &[A]) -> Vec<A> {
    let mut res = unsafe { uninitialized_vec(vec.len()) };
    parallel_clone(vec, &mut res);
    res
}

pub fn dot_product_ef_packed_par<EF: ExtensionField<PF<EF>>, R: Sync + Send + Copy>(
    a: &[EFPacking<EF>],
    b: &[R],
) -> EF
where
    EFPacking<EF>: Algebra<R>,
{
    assert_eq!(a.len(), b.len());
    let res_packed: EFPacking<EF> = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| x * y)
        .sum::<EFPacking<EF>>();
    unpack_extension(&[res_packed]).into_iter().sum()
}

pub fn split_at_mut_many<'a, A>(slice: &'a mut [A], indices: &[usize]) -> Vec<&'a mut [A]> {
    for i in 0..indices.len() {
        if i > 0 {
            assert!(indices[i] > indices[i - 1]);
        }
        assert!(indices[i] <= slice.len());
    }

    if indices.is_empty() {
        return vec![slice];
    }

    let mut result = Vec::with_capacity(indices.len() + 1);
    let mut current_slice = slice;
    let mut prev_idx = 0;

    for &idx in indices {
        let adjusted_idx = idx - prev_idx;
        let (left, right) = current_slice.split_at_mut(adjusted_idx);
        result.push(left);
        current_slice = right;
        prev_idx = idx;
    }

    result.push(current_slice);

    result
}

// pub fn convert_array<A, const N: usize, const M: usize>(input: [A; N]) -> [A; M] {
//     assert_eq!(N, M);
//     unsafe {
//         let output = std::ptr::read(&input as *const [A; N] as *const [A; M]);
//         std::mem::forget(input);
//         output
//     }
// }

#[cfg(test)]
mod tests {
    use p3_koala_bear::{KoalaBear, QuinticExtensionFieldKB};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    type F = KoalaBear;
    type EF = QuinticExtensionFieldKB;

    #[test]
    fn test_dot_product_ef_f_packed() {
        let n = 1 << 20;
        let mut rng = StdRng::seed_from_u64(0);
        let a: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let b: Vec<F> = (0..n).map(|_| rng.random()).collect();
        assert_eq!(
            dot_product_ef_packed_par::<EF, _>(
                &pack_extension(&a),
                PFPacking::<EF>::pack_slice(&b)
            ),
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| x * y)
                .sum::<EF>()
        );
    }

    #[test]
    fn test_dot_product_ef_ef_packed() {
        let n = 1 << 20;
        let mut rng = StdRng::seed_from_u64(0);
        let a: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let b: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        assert_eq!(
            dot_product_ef_packed_par::<EF, _>(&pack_extension(&a), &pack_extension(&b),),
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| x * y)
                .sum::<EF>()
        );
    }
}
