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

pub fn batch_fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
    polys: &[&[F]],
    scalars: &[EF],
) -> Vec<Vec<EF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear_in_large_field(poly, scalars))
        .collect()
}

pub fn batch_fold_multilinear_in_large_field_packed<EF: ExtensionField<PF<EF>>>(
    polys: &[&[EFPacking<EF>]],
    scalars: &[EF],
) -> Vec<Vec<EFPacking<EF>>> {
    polys
        .iter()
        .map(|poly| fold_extension_packed(poly, scalars))
        .collect()
}

pub fn fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
    m: &[F],
    scalars: &[EF],
) -> Vec<EF> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.len());
    let new_size = m.len() / scalars.len();
    (0..new_size)
        .into_par_iter()
        .map(|i| {
            scalars
                .iter()
                .enumerate()
                .map(|(j, s)| *s * m[i + j * new_size])
                .sum()
        })
        .collect()
}

pub fn fold_extension_packed<EF: ExtensionField<PF<EF>>>(
    m: &[EFPacking<EF>],
    scalars: &[EF],
) -> Vec<EFPacking<EF>> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.len());
    let new_size = m.len() / scalars.len();

    (0..new_size)
        .into_par_iter()
        .map(|i| {
            scalars
                .iter()
                .enumerate()
                .map(|(j, s)| m[i + j * new_size] * *s)
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
