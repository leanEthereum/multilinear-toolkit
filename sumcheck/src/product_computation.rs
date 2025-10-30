use std::any::TypeId;

use backend::{
    DensePolynomial, MleGroupOwned, MleOwned, MleRef, MultilinearPoint, par_zip_fold_2,
    uninitialized_vec,
};
use fiat_shamir::*;
use p3_field::*;
use rayon::prelude::*;

use crate::{SumcheckComputation, SumcheckComputationPacked, sumcheck_prove_many_rounds};

#[derive(Debug)]
pub struct ProductComputation;

impl<IF: ExtensionField<PF<EF>>, EF: ExtensionField<IF>> SumcheckComputation<IF, EF>
    for ProductComputation
{
    fn eval(&self, point: &[IF], _: &[EF]) -> EF {
        if TypeId::of::<IF>() == TypeId::of::<EF>() {
            let point = unsafe { std::mem::transmute::<&[IF], &[EF]>(point) };
            unsafe { *point.get_unchecked(0) * *point.get_unchecked(1) }
        } else {
            todo!("There would be embedding overhead ...?")
        }
    }
    fn degree(&self) -> usize {
        2
    }
}

impl<EF: ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for ProductComputation {
    fn eval_packed_base(&self, point: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        // TODO this is very inneficient
        EFPacking::<EF>::from(point[0] * point[1])
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        unsafe { *point.get_unchecked(0) * *point.get_unchecked(1) }
    }
    fn degree(&self) -> usize {
        2
    }
}

pub fn run_product_sumcheck<EF: ExtensionField<PF<EF>>>(
    pol_a: &MleRef<'_, EF>, // evals
    pol_b: &MleRef<'_, EF>, // weights
    prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
    mut sum: EF,
    n_rounds: usize,
) -> (MultilinearPoint<EF>, EF, MleOwned<EF>, MleOwned<EF>) {
    assert!(n_rounds >= 1);
    let first_sumcheck_poly = match (pol_a, pol_b) {
        (MleRef::BasePacked(evals), MleRef::ExtensionPacked(weights)) => {
            compute_product_sumcheck_polynomial(&evals, &weights, sum, |e| {
                EFPacking::<EF>::to_ext_iter([e]).collect()
            })
        }
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            compute_product_sumcheck_polynomial(evals, &weights, sum, |e| {
                EFPacking::<EF>::to_ext_iter([e]).collect()
            })
        }
        (MleRef::Base(evals), MleRef::Extension(weights)) => {
            compute_product_sumcheck_polynomial(evals, &weights, sum, |e| vec![e])
        }
        (MleRef::Extension(evals), MleRef::Extension(weights)) => {
            compute_product_sumcheck_polynomial(evals, &weights, sum, |e| vec![e])
        }
        _ => unimplemented!(),
    };

    prover_state.add_extension_scalars(&first_sumcheck_poly.coeffs);
    // TODO: re-enable PoW grinding
    let r1: EF = prover_state.sample();
    sum = first_sumcheck_poly.evaluate(r1);

    if n_rounds == 1 {
        return (
            MultilinearPoint(vec![r1]),
            sum,
            pol_a.fold(&[EF::ONE - r1, r1]),
            pol_b.fold(&[EF::ONE - r1, r1]),
        );
    }

    let (second_sumcheck_poly, folded) = match (pol_a, pol_b) {
        (MleRef::BasePacked(evals), MleRef::ExtensionPacked(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(&evals, &weights, r1, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                });
            (second_sumcheck_poly, MleGroupOwned::ExtensionPacked(folded))
        }
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, &weights, r1, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                });
            (second_sumcheck_poly, MleGroupOwned::ExtensionPacked(folded))
        }
        (MleRef::Base(evals), MleRef::Extension(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, &weights, r1, sum, |e| vec![e]);
            (second_sumcheck_poly, MleGroupOwned::Extension(folded))
        }
        (MleRef::Extension(evals), MleRef::Extension(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, &weights, r1, sum, |e| vec![e]);
            (second_sumcheck_poly, MleGroupOwned::Extension(folded))
        }
        _ => unimplemented!(),
    };

    prover_state.add_extension_scalars(&second_sumcheck_poly.coeffs);
    // TODO: re-enable PoW grinding
    let r2: EF = prover_state.sample();
    sum = second_sumcheck_poly.evaluate(r2);

    let (mut challenges, folds, sum) = sumcheck_prove_many_rounds(
        1,
        folded,
        Some(vec![EF::ONE - r2, r2]),
        &ProductComputation,
        &[],
        None,
        false,
        prover_state,
        sum,
        None,
        n_rounds - 2,
    );

    challenges.splice(0..0, [r1, r2]);
    let [pol_a, pol_b] = folds.split().try_into().unwrap();
    (challenges, sum, pol_a, pol_b)
}

pub fn compute_product_sumcheck_polynomial<
    F: PrimeCharacteristicRing + Copy + Send + Sync,
    EF: Field,
    EFPacking: Algebra<F> + Copy + Send + Sync,
>(
    pol_0: &[F],         // evals
    pol_1: &[EFPacking], // weights
    sum: EF,
    decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> DensePolynomial<EF> {
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());

    let (c0_packed, c2_packed) = pol_0[..n / 2]
        .par_iter()
        .zip(pol_0[n / 2..].par_iter())
        .zip(pol_1[..n / 2].par_iter().zip(pol_1[n / 2..].par_iter()))
        .map(sumcheck_quadratic)
        .reduce(
            || (EFPacking::ZERO, EFPacking::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    let c0 = decompose(c0_packed).into_iter().sum::<EF>();
    let c2 = decompose(c2_packed).into_iter().sum::<EF>();

    let c1 = sum - c0.double() - c2;

    DensePolynomial::new(vec![c0, c1, c2])
}

pub fn fold_and_compute_product_sumcheck_polynomial<
    F: PrimeCharacteristicRing + Copy + Send + Sync,
    EF: Field,
    EFPacking: Algebra<F> + From<EF> + Copy + Send + Sync,
>(
    pol_0: &[F],         // evals
    pol_1: &[EFPacking], // weights
    prev_folding_factor: EF,
    sum: EF,
    decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> (DensePolynomial<EF>, Vec<Vec<EFPacking>>) {
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());
    let prev_folding_factor_packed = EFPacking::from(prev_folding_factor);

    let mut pol_0_folded = unsafe { uninitialized_vec::<EFPacking>(n / 2) };
    let mut pol_1_folded = unsafe { uninitialized_vec::<EFPacking>(n / 2) };

    let (c0_packed, c2_packed) = par_zip_fold_2(pol_0, &mut pol_0_folded)
        .zip(par_zip_fold_2(pol_1, &mut pol_1_folded))
        .map(|((p0_prev, p0_f), (p1_prev, p1_f))| {
            let pol_0_folded_left =
                prev_folding_factor_packed * (*p0_prev.1.0 - *p0_prev.0.0) + *p0_prev.0.0;
            let pol_0_folded_right =
                prev_folding_factor_packed * (*p0_prev.1.1 - *p0_prev.0.1) + *p0_prev.0.1;
            *p0_f.0 = pol_0_folded_left;
            *p0_f.1 = pol_0_folded_right;

            let pol_1_folded_left =
                prev_folding_factor_packed * (*p1_prev.1.0 - *p1_prev.0.0) + *p1_prev.0.0;
            let pol_1_folded_right =
                prev_folding_factor_packed * (*p1_prev.1.1 - *p1_prev.0.1) + *p1_prev.0.1;
            *p1_f.0 = pol_1_folded_left;
            *p1_f.1 = pol_1_folded_right;

            sumcheck_quadratic((
                (&pol_0_folded_left, &pol_0_folded_right),
                (&pol_1_folded_left, &pol_1_folded_right),
            ))
        })
        .reduce(
            || (EFPacking::ZERO, EFPacking::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    let c0 = decompose(c0_packed).into_iter().sum::<EF>();
    let c2 = decompose(c2_packed).into_iter().sum::<EF>();

    let c1 = sum - c0.double() - c2;

    (
        DensePolynomial::new(vec![c0, c1, c2]),
        vec![pol_0_folded, pol_1_folded],
    )
}

#[inline]
pub fn sumcheck_quadratic<F, EF>(((&x_0, &x_1), (&y_0, &y_1)): ((&F, &F), (&EF, &EF))) -> (EF, EF)
where
    F: PrimeCharacteristicRing + Copy,
    EF: Algebra<F> + Copy,
{
    // Compute the constant coefficient:
    // p(0) * w(0)
    let constant = y_0 * x_0;

    // Compute the quadratic coefficient:
    // (p(1) - p(0)) * (w(1) - w(0))
    let quadratic = (y_1 - y_0) * (x_1 - x_0);

    (constant, quadratic)
}
