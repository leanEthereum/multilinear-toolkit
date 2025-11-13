use std::any::TypeId;

use fiat_shamir::*;
use p3_field::*;

use crate::{SumcheckComputation, SumcheckComputationPacked};

#[derive(Debug)]
pub struct CubeComputation;

impl<IF: ExtensionField<PF<EF>>, EF: ExtensionField<IF>> SumcheckComputation<IF, EF>
    for CubeComputation
{
    fn eval(&self, point: &[IF], _: &[EF]) -> EF {
        if TypeId::of::<IF>() == TypeId::of::<EF>() {
            let point = unsafe { std::mem::transmute::<&[IF], &[EF]>(point) };
            point[0] * point[1] * point[2]
        } else {
            todo!("There would be embedding overhead ...?")
        }
    }
    fn degree(&self) -> usize {
        3
    }
}

impl<EF: ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for CubeComputation {
    fn eval_packed_base(&self, _point: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        todo!("There would be embedding overhead ...?")
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        point[0] * point[1] * point[2]
    }
    fn degree(&self) -> usize {
        3
    }
}
