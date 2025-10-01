use std::any::TypeId;

use backend::{DensePolynomial, Mle, MleGroupOwned, MleRef, MultilinearPoint};
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
    fn eval_packed_base(&self, _: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        unreachable!()
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        unsafe { *point.get_unchecked(0) * *point.get_unchecked(1) }
    }
    fn degree(&self) -> usize {
        2
    }
}

pub fn run_product_sumcheck<EF: ExtensionField<PF<EF>>>(
    pol_a: &mut Mle<'_, EF>,
    pol_b: &mut Mle<'_, EF>,
    prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
    mut sum: EF,
    n_rounds: usize,
) -> (MultilinearPoint<EF>, EF) {
    assert!(n_rounds >= 1);
    let sumcheck_poly = match (pol_a.by_ref(), pol_b.by_ref()) {
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
        _ => unimplemented!(),
    };
    prover_state.add_extension_scalars(&sumcheck_poly.coeffs);

    // TODO: re-enable PoW grinding
    // prover_state.pow_grinding(pow_bits);

    let r: EF = prover_state.sample();

    pol_b.fold_in_place(&[(EF::ONE - r), r]);
    pol_a.fold_in_place(&[(EF::ONE - r), r]);

    sum = sumcheck_poly.evaluate(r);

    let (mut challenges, folds, sum) = sumcheck_prove_many_rounds(
        1,
        MleGroupOwned::merge(vec![
            std::mem::take(&mut pol_a.as_owned_mut().unwrap()),
            std::mem::take(&mut pol_b.as_owned_mut().unwrap()),
        ]),
        &ProductComputation,
        &ProductComputation,
        &[],
        None,
        false,
        prover_state,
        sum,
        None,
        n_rounds - 1,
    );

    let [evals_folded, weights_folded] = folds.as_owned().unwrap().split().try_into().unwrap();
    *pol_a = Mle::Owned(evals_folded);
    *pol_b = Mle::Owned(weights_folded);

    challenges.insert(0, r);
    (challenges, sum)
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

#[inline]
pub(crate) fn sumcheck_quadratic<F, EF>(
    ((&x_0, &x_1), (&y_0, &y_1)): ((&F, &F), (&EF, &EF)),
) -> (EF, EF)
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
