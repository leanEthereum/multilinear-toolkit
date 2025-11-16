use std::ops::{Add, Mul};

use fiat_shamir::{EFPacking, PF, PFPacking};
use p3_field::ExtensionField;

use crate::{SumcheckComputation, mul_many_const};

#[derive(Default, Debug)]
pub struct GKRQuotientComputation<const N: usize>;

impl<const N: usize, EF: ExtensionField<PF<EF>>> SumcheckComputation<EF>
    for GKRQuotientComputation<N>
{
    fn degree(&self) -> usize {
        N
    }

    #[inline(always)]
    fn eval_base(&self, point: &[PF<EF>], alphas: &[EF]) -> EF {
        let num = numerator_of_sum_of_quotients::<N, _>(&point[..N], &point[N..]);
        let denom = mul_many_const::<N, _>(&point[N..]);
        alphas[0] * denom + num
    }

    #[inline(always)]
    fn eval_extension(&self, point: &[EF], alphas: &[EF]) -> EF {
        let num = numerator_of_sum_of_quotients::<N, _>(&point[..N], &point[N..]);
        let denom = mul_many_const::<N, _>(&point[N..]);
        alphas[0] * denom + num
    }

    #[inline(always)]
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alphas: &[EF]) -> EFPacking<EF> {
        let num = numerator_of_sum_of_quotients::<N, _>(&point[..N], &point[N..]);
        let denom = mul_many_const::<N, _>(&point[N..]);
        EFPacking::<EF>::from(alphas[0]) * denom + num
    }

    #[inline(always)]
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alphas: &[EF]) -> EFPacking<EF> {
        let num = numerator_of_sum_of_quotients::<N, _>(&point[..N], &point[N..]);
        let denom = mul_many_const::<N, _>(&point[N..]);
        EFPacking::<EF>::from(alphas[0]) * denom + num
    }
}

#[inline(always)]
pub fn numerator_of_sum_of_quotients<
    const N: usize,
    A: Copy + Mul<Output = A> + Add<Output = A>,
>(
    numerators: &[A],
    denominators: &[A],
) -> A {
    debug_assert_eq!(numerators.len(), N);
    debug_assert_eq!(denominators.len(), N);
    match N {
        2 => numerators[0] * denominators[1] + numerators[1] * denominators[0],
        4 => {
            let denom_1_2 = denominators[1] * denominators[2];
            let denom_0_3 = denominators[0] * denominators[3];
            numerators[0] * denom_1_2 * denominators[3]
                + numerators[1] * denominators[2] * denom_0_3
                + numerators[2] * denominators[1] * denom_0_3
                + numerators[3] * denominators[0] * denom_1_2
        }
        8 => {
            let d01 = denominators[0] * denominators[1];
            let d23 = denominators[2] * denominators[3];
            let d45 = denominators[4] * denominators[5];
            let d67 = denominators[6] * denominators[7];

            let d0123 = d01 * d23;
            let d4567 = d45 * d67;

            let d1234567 = denominators[1] * d23 * d4567;
            let d0234567 = denominators[0] * d23 * d4567;
            let d0134567 = denominators[1] * denominators[0] * denominators[3] * d4567;
            let d0124567 = d01 * denominators[2] * d4567;
            let d0123567 = d0123 * denominators[5] * d67;
            let d0123467 = d0123 * denominators[4] * d67;
            let d0123457 = d0123 * d45 * denominators[7];
            let d0123456 = d0123 * d45 * denominators[6];

            numerators[0] * d1234567
                + numerators[1] * d0234567
                + numerators[2] * d0134567
                + numerators[3] * d0124567
                + numerators[4] * d0123567
                + numerators[5] * d0123467
                + numerators[6] * d0123457
                + numerators[7] * d0123456
        }
        _ => unimplemented!(),
    }
}
