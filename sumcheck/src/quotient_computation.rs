use fiat_shamir::{EFPacking, PF, PFPacking};
use p3_field::{ExtensionField, Field};

use crate::{SumcheckComputation, SumcheckComputationPacked};

#[derive(Default, Debug)]
pub struct GKRQuotientComputation;

impl<IF: ExtensionField<PF<EF>>, EF: ExtensionField<IF>> SumcheckComputation<IF, EF>
    for GKRQuotientComputation
{
    fn eval(&self, point: &[IF], alphas: &[EF]) -> EF {
        let num = point[0] * point[3] + point[1] * point[2];
        let denom = point[2] * point[3];
        alphas[0] * denom + num
    }
    fn degree(&self) -> usize {
        2
    }
}

impl<EF: ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for GKRQuotientComputation {
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alphas: &[EF]) -> EFPacking<EF> {
        let num = point[0] * point[3] + point[1] * point[2];
        let denom = point[2] * point[3];
        EFPacking::<EF>::from(alphas[0]) * denom + num
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alphas: &[EF]) -> EFPacking<EF> {
        let num = point[0] * point[3] + point[1] * point[2];
        let denom = point[2] * point[3];
        EFPacking::<EF>::from(alphas[0]) * denom + num
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
