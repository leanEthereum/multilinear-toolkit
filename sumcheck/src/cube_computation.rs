use fiat_shamir::*;
use p3_field::*;

use crate::{SumcheckComputation, SumcheckComputationPacked};

#[derive(Debug)]
pub struct CubeComputation;

impl<IF: ExtensionField<PF<EF>>, EF: ExtensionField<IF>> SumcheckComputation<IF, EF>
    for CubeComputation
{
    fn eval(&self, point: &[IF], _: &[EF]) -> EF {
        // TODO avoid embedding overhead
        EF::from(point[0].cube())
    }
    fn degree(&self) -> usize {
        3
    }
}

impl<EF: ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for CubeComputation {
    fn eval_packed_base(&self, point: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        // TODO avoid embedding overhead
        EFPacking::<EF>::from(point[0].cube())
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        point[0].cube()
    }
    fn degree(&self) -> usize {
        3
    }
}
