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
    vec.iter()
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
        .iter()
        .map(|poly| fold_multilinear(poly, scalars, &mul_if_of))
        .collect()
}

pub fn batch_fold_multilinear_in_place<F: Field, NF: Algebra<F> + Sync + Send + Copy>(
    polys: &mut [&mut Vec<NF>],
    scalars: &[F],
) {
    polys
        .iter_mut()
        .for_each(|poly| fold_multilinear_in_place(poly, scalars));
}

pub fn fold_multilinear_in_place<F: Field, NF: Algebra<F> + Sync + Send + Copy>(
    m: &mut Vec<NF>,
    scalars: &[F],
) {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.len());
    let new_size = m.len() / scalars.len();
    let (left, right) = m.split_at_mut(new_size);

    if scalars.len() == 2 {
        assert_eq!(scalars[0], F::ONE - scalars[1]);
        let alpha = scalars[1];
        left.par_iter_mut().enumerate().for_each(|(i, out)| {
            let s = (right[i] - *out) * alpha + *out;
            *out = s;
        });
    } else {
        left.par_iter_mut().enumerate().for_each(|(i, out)| {
            let s = *out * scalars[0]
                + scalars
                    .iter()
                    .skip(1)
                    .enumerate()
                    .map(|(j, s)| right[j * new_size + i] * *s) // only reads
                    .sum::<NF>();
            *out = s;
        });
    }

    m.truncate(new_size);
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
