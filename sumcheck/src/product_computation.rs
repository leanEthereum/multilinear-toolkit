use std::any::TypeId;

use backend::{DensePolynomial, MleGroupOwned, MleOwned, MleRef, MultilinearPoint};
use fiat_shamir::*;
use p3_field::*;

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
    _pol_0: &[F],         // evals
    _pol_1: &[EFPacking], // weights
    _sum: EF,
    _decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> DensePolynomial<EF> {
    unimplemented!()
}

pub fn fold_and_compute_product_sumcheck_polynomial<
    F: PrimeCharacteristicRing + Copy + Send + Sync,
    EF: Field,
    EFPacking: Algebra<F> + From<EF> + Copy + Send + Sync,
>(
    _pol_0: &[F],         // evals
    _pol_1: &[EFPacking], // weights
    _prev_folding_factor: EF,
    _sum: EF,
    _decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> (DensePolynomial<EF>, Vec<Vec<EFPacking>>) {
    unimplemented!()
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
