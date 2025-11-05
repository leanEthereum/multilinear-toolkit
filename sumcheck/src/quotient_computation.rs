use backend::DensePolynomial;
use fiat_shamir::{EFPacking, PF, PFPacking};
use p3_field::{Algebra, ExtensionField, Field};

use crate::{SumcheckComputation, SumcheckComputationPacked};

pub struct GKRQuotientComputation<EF> {
    pub u4_const: EF,
    pub u5_const: EF,
}

impl<IF: ExtensionField<PF<EF>>, EF: ExtensionField<IF>> SumcheckComputation<IF, EF>
    for GKRQuotientComputation<EF>
{
    fn eval(&self, point: &[IF], _: &[EF]) -> EF {
        // U4.U2.U3 + U5.[U0.U3 + U1.U2]
        self.u4_const * point[2] * point[3]
            + self.u5_const * (point[0] * point[3] + point[1] * point[2])
    }
    fn degree(&self) -> usize {
        2
    }
}

impl<EF: ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for GKRQuotientComputation<EF> {
    fn eval_packed_base(&self, _: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        todo!()
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        point[2] * point[3] * self.u4_const
            + (point[0] * point[3] + point[1] * point[2]) * self.u5_const
    }
    fn degree(&self) -> usize {
        2
    }
}

pub struct GKRQuotientCrossComputation {}

impl<IF: Field, EF: ExtensionField<IF>> SumcheckComputation<IF, EF>
    for GKRQuotientCrossComputation
{
    fn eval(&self, point: &[IF], _: &[EF]) -> EF {
        EF::from(point[0] * point[3] + point[1] * point[2])
    }
    fn degree(&self) -> usize {
        2
    }
}

impl<EF: ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for GKRQuotientCrossComputation {
    fn eval_packed_base(&self, _: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        todo!()
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        point[0] * point[3] + point[1] * point[2]
    }
    fn degree(&self) -> usize {
        2
    }
}

pub fn compute_gkr_quotient_sumcheck_polynomial<F: Algebra<EF> + Copy + Send + Sync, EF: Field>(
    _u0: &[F],
    _u1: &[F],
    _u2: &[F],
    _u3: &[F],
    _u4_const: EF,
    _u5_const: EF,
    _first_eq_factor: EF,
    _eq_mle: &[F],
    _missing_mul_factor: EF,
    _sum: EF,
    _decompose: impl Fn(F) -> Vec<EF>,
) -> DensePolynomial<EF> {
    todo!()
}

pub fn fold_and_compute_gkr_quotient_sumcheck_polynomial<
    F: Algebra<EF> + Copy + Send + Sync,
    EF: Field,
>(
    _prev_folding_factor: EF,
    _u0: &[F],
    _u1: &[F],
    _u2: &[F],
    _u3: &[F],
    _u4_const: EF,
    _u5_const: EF,
    _first_eq_factor: EF,
    _eq_mle: &[F],
    _missing_mul_factor: EF,
    _sum: EF,
    _decompose: impl Fn(F) -> Vec<EF>,
) -> (DensePolynomial<EF>, Vec<Vec<F>>) {
    unimplemented!()
}
