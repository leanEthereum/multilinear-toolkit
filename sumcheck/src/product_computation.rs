use std::any::TypeId;

use backend::DensePolynomial;
use fiat_shamir::*;
use p3_field::*;
use rayon::prelude::*;

use crate::{SumcheckComputation, SumcheckComputationPacked};

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
