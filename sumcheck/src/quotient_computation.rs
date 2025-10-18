use backend::{DensePolynomial, par_iter_split_2, par_zip_fold_2, uninitialized_vec};
use fiat_shamir::{EFPacking, PF, PFPacking};
use p3_field::{Algebra, ExtensionField, Field};
use rayon::prelude::*;

use crate::{SumcheckComputation, SumcheckComputationPacked, sumcheck_quadratic};

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
    u0: &[F],
    u1: &[F],
    u2: &[F],
    u3: &[F],
    u4_const: EF,
    u5_const: EF,
    first_eq_factor: EF,
    eq_mle: &[F],
    missing_mul_factor: EF,
    sum: EF,
    decompose: impl Fn(F) -> Vec<EF>,
) -> DensePolynomial<EF> {
    let n = u0.len();
    assert_eq!(eq_mle.len(), n / 2);

    let (c0_term_single, c2_term_single, c0_term_double, c2_term_double) = par_iter_split_2(u0)
        .zip(par_iter_split_2(u1))
        .zip(par_iter_split_2(u2))
        .zip(par_iter_split_2(u3))
        .zip(eq_mle.par_iter())
        .map(
            |(
                (
                    (((u0_left, u0_right), (u1_left, u1_right)), (u2_left, u2_right)),
                    (u3_left, u3_right),
                ),
                &eq_val,
            )| {
                let (mut c0_term_single, mut c2_term_single) =
                    sumcheck_quadratic(((u2_left, u2_right), (u3_left, u3_right)));
                c0_term_single *= eq_val;
                c2_term_single *= eq_val;

                let (c0_term_double_a, c2_term_double_a) =
                    sumcheck_quadratic(((u0_left, u0_right), (u3_left, u3_right)));
                let (c0_term_double_b, c2_term_double_b) =
                    sumcheck_quadratic(((u1_left, u1_right), (u2_left, u2_right)));
                let mut c0_term_double = c0_term_double_a + c0_term_double_b;
                let mut c2_term_double = c2_term_double_a + c2_term_double_b;
                c0_term_double *= eq_val;
                c2_term_double *= eq_val;

                (
                    c0_term_single,
                    c2_term_single,
                    c0_term_double,
                    c2_term_double,
                )
            },
        )
        .reduce(
            || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
            |(a0, a1, a2, a3), (b0, b1, b2, b3)| (a0 + b0, a1 + b1, a2 + b2, a3 + b3),
        );

    let c0 = c0_term_single * u4_const + c0_term_double * u5_const;
    let c2 = c2_term_single * u4_const + c2_term_double * u5_const;

    let c0 = decompose(c0).into_iter().sum::<EF>();
    let c2 = decompose(c2).into_iter().sum::<EF>();

    let c1 = ((sum / missing_mul_factor) - c2 * first_eq_factor - c0) / first_eq_factor;

    DensePolynomial::new(vec![
        c0 * missing_mul_factor,
        c1 * missing_mul_factor,
        c2 * missing_mul_factor,
    ])
}

pub fn fold_and_compute_gkr_quotient_sumcheck_polynomial<
    F: Algebra<EF> + Copy + Send + Sync,
    EF: Field,
>(
    prev_folding_factor: EF,
    u0: &[F],
    u1: &[F],
    u2: &[F],
    u3: &[F],
    u4_const: EF,
    u5_const: EF,
    first_eq_factor: EF,
    eq_mle: &[F],
    missing_mul_factor: EF,
    sum: EF,
    decompose: impl Fn(F) -> Vec<EF>,
) -> (DensePolynomial<EF>, Vec<Vec<F>>) {
    let n = u0.len();
    assert_eq!(eq_mle.len(), n / 4);

    let mut folded_u0 = unsafe { uninitialized_vec::<F>(n / 2) };
    let mut folded_u1 = unsafe { uninitialized_vec::<F>(n / 2) };
    let mut folded_u2 = unsafe { uninitialized_vec::<F>(n / 2) };
    let mut folded_u3 = unsafe { uninitialized_vec::<F>(n / 2) };

    let my_fold = |u: ((&F, &F), (&F, &F)), folded: (&mut F, &mut F)| {
        let u_left = *u.0.0 + (*u.1.0 - *u.0.0) * prev_folding_factor;
        let u_right = *u.0.1 + (*u.1.1 - *u.0.1) * prev_folding_factor;
        *folded.0 = u_left;
        *folded.1 = u_right;
        (u_left, u_right)
    };

    let (c0_term_single, c2_term_single, c0_term_double, c2_term_double) =
        par_zip_fold_2(u0, &mut folded_u0)
            .zip(par_zip_fold_2(u1, &mut folded_u1))
            .zip(par_zip_fold_2(u2, &mut folded_u2))
            .zip(par_zip_fold_2(u3, &mut folded_u3))
            .zip(eq_mle.par_iter())
            .map(
                |(
                    ((((u0_prev, u0_f), (u1_prev, u1_f)), (u2_prev, u2_f)), (u3_prev, u3_f)),
                    &eq_val,
                )| {
                    let (u0_left, u0_right) = my_fold(u0_prev, u0_f);
                    let (u1_left, u1_right) = my_fold(u1_prev, u1_f);
                    let (u2_left, u2_right) = my_fold(u2_prev, u2_f);
                    let (u3_left, u3_right) = my_fold(u3_prev, u3_f);

                    let (mut c0_term_single, mut c2_term_single) =
                        sumcheck_quadratic(((&u2_left, &u2_right), (&u3_left, &u3_right)));
                    c0_term_single *= eq_val;
                    c2_term_single *= eq_val;

                    let (c0_term_double_a, c2_term_double_a) =
                        sumcheck_quadratic(((&u0_left, &u0_right), (&u3_left, &u3_right)));
                    let (c0_term_double_b, c2_term_double_b) =
                        sumcheck_quadratic(((&u1_left, &u1_right), (&u2_left, &u2_right)));
                    let mut c0_term_double = c0_term_double_a + c0_term_double_b;
                    let mut c2_term_double = c2_term_double_a + c2_term_double_b;
                    c0_term_double *= eq_val;
                    c2_term_double *= eq_val;

                    (
                        c0_term_single,
                        c2_term_single,
                        c0_term_double,
                        c2_term_double,
                    )
                },
            )
            .reduce(
                || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
                |(a0, a1, a2, a3), (b0, b1, b2, b3)| (a0 + b0, a1 + b1, a2 + b2, a3 + b3),
            );

    let c0 = c0_term_single * u4_const + c0_term_double * u5_const;
    let c2 = c2_term_single * u4_const + c2_term_double * u5_const;

    let c0 = decompose(c0).into_iter().sum::<EF>();
    let c2 = decompose(c2).into_iter().sum::<EF>();

    let c1 = ((sum / missing_mul_factor) - c2 * first_eq_factor - c0) / first_eq_factor;

    (
        DensePolynomial::new(vec![
            c0 * missing_mul_factor,
            c1 * missing_mul_factor,
            c2 * missing_mul_factor,
        ]),
        vec![folded_u0, folded_u1, folded_u2, folded_u3],
    )
}
