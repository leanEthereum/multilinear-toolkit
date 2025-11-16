use fiat_shamir::{EFPacking, PF, PFPacking};
use p3_field::ExtensionField;

use crate::SumcheckComputation;

#[derive(Default, Debug)]
pub struct GKRQuotientComputation;

impl<EF: ExtensionField<PF<EF>>> SumcheckComputation<EF> for GKRQuotientComputation {
    fn degree(&self) -> usize {
        2
    }
    fn eval_base(&self, point: &[PF<EF>], alphas: &[EF]) -> EF {
        let num = point[0] * point[3] + point[1] * point[2];
        let denom = point[2] * point[3];
        alphas[0] * denom + num
    }

    fn eval_extension(&self, point: &[EF], alphas: &[EF]) -> EF {
        let num = point[0] * point[3] + point[1] * point[2];
        let denom = point[2] * point[3];
        alphas[0] * denom + num
    }

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
}
