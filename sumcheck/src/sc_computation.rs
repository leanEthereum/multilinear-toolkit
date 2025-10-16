use crate::*;
use backend::*;
use constraints_folder::*;
use fiat_shamir::*;
use p3_air::Air;
use p3_field::PackedFieldExtension;
use p3_field::PrimeCharacteristicRing;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use rayon::prelude::*;
use std::any::TypeId;
use std::ops::Mul;

pub trait SumcheckComputation<NF, EF>: Sync {
    fn degree(&self) -> usize;
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF;
}

impl<NF, EF, A> SumcheckComputation<NF, EF> for A
where
    NF: ExtensionField<PF<EF>>,
    EF: ExtensionField<NF> + ExtensionField<PF<EF>>,
    A: for<'a> Air<ConstraintFolder<'a, NF, EF>>,
{
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF {
        if self.structured() {
            assert_eq!(point.len(), A::width(self) * 2);
        } else {
            assert_eq!(point.len(), A::width(self));
        }
        let mut folder = ConstraintFolder {
            main: RowMajorMatrixView::new(point, A::width(self)),
            alpha_powers,
            accumulator: EF::ZERO,
            constraint_index: 0,
        };
        self.eval(&mut folder);
        folder.accumulator
    }
    fn degree(&self) -> usize {
        self.degree()
    }
}

pub trait SumcheckComputationPacked<EF>: Sync
where
    EF: ExtensionField<PF<EF>>,
{
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF>;
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF>;
    fn degree(&self) -> usize;
}

impl<EF: Field, A> SumcheckComputationPacked<EF> for A
where
    EF: ExtensionField<PF<EF>>,
    A: for<'a> Air<ConstraintFolderPackedBase<'a, EF>>
        + for<'a> Air<ConstraintFolderPackedExtension<'a, EF>>,
{
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF> {
        if self.structured() {
            assert_eq!(point.len(), A::width(self) * 2);
        } else {
            assert_eq!(point.len(), A::width(self));
        }
        let mut folder = ConstraintFolderPackedBase {
            main: RowMajorMatrixView::new(point, A::width(self)),
            alpha_powers,
            accumulator: Default::default(),
            constraint_index: 0,
        };
        self.eval(&mut folder);

        folder.accumulator
    }

    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF> {
        if self.structured() {
            assert_eq!(point.len(), A::width(self) * 2);
        } else {
            assert_eq!(point.len(), A::width(self));
        }
        let mut folder = ConstraintFolderPackedExtension {
            main: RowMajorMatrixView::new(point, A::width(self)),
            alpha_powers,
            accumulator: Default::default(),
            constraint_index: 0,
        };
        self.eval(&mut folder);

        folder.accumulator
    }

    fn degree(&self) -> usize {
        self.degree()
    }
}

pub fn sumcheck_compute<'a, EF: ExtensionField<PF<EF>>, SC, SCP>(
    group: &MleGroupRef<'a, EF>,
    params: SumcheckComputeParams<'a, EF, SC, SCP>,
    zs: &[usize],
) -> Vec<(PF<EF>, EF)>
where
    SC: SumcheckComputation<PF<EF>, EF> + SumcheckComputation<EF, EF> + 'static,
    SCP: SumcheckComputationPacked<EF>,
{
    let SumcheckComputeParams {
        skips,
        eq_mle,
        folding_factors,
        computation,
        computation_packed,
        first_eq_factor,
        batching_scalars,
        missing_mul_factor,
        sum,
    } = params;

    let fold_size = 1 << (group.n_vars() - skips);
    let packed_fold_size = if group.is_packed() {
        fold_size / packing_width::<EF>()
    } else {
        fold_size
    };

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<ProductComputation>() && eq_mle.is_none() {
        assert!(missing_mul_factor.is_none());
        assert!(batching_scalars.is_empty());
        assert_eq!(group.n_columns(), 2);

        let poly = match group {
            MleGroupRef::Extension(multilinears) => {
                compute_product_sumcheck_polynomial(&multilinears[0], &multilinears[1], sum, |e| {
                    vec![e]
                })
            }
            MleGroupRef::ExtensionPacked(multilinears) => {
                compute_product_sumcheck_polynomial(&multilinears[0], &multilinears[1], sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                })
            }
            _ => unimplemented!(),
        };
        return vec![
            (PF::<EF>::ZERO, poly.coeffs[0]),
            (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
        ];
    }

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<EF>>() {
        assert!(eq_mle.is_some());
        assert!(batching_scalars.is_empty());
        assert_eq!(group.n_columns(), 4);

        let sc_computation =
            unsafe { std::mem::transmute::<&SC, &GKRQuotientComputation<EF>>(computation) };

        let poly = match group {
            MleGroupRef::Extension(multilinears) => compute_gkr_quotient_sumcheck_polynomial(
                &multilinears[0],
                &multilinears[1],
                &multilinears[2],
                &multilinears[3],
                sc_computation.u4_const,
                sc_computation.u5_const,
                first_eq_factor.unwrap(),
                eq_mle.unwrap().as_extension().unwrap(),
                missing_mul_factor.unwrap_or(EF::ONE),
                sum,
                |e| vec![e],
            ),
            MleGroupRef::ExtensionPacked(multilinears) => compute_gkr_quotient_sumcheck_polynomial(
                &multilinears[0],
                &multilinears[1],
                &multilinears[2],
                &multilinears[3],
                sc_computation.u4_const,
                sc_computation.u5_const,
                first_eq_factor.unwrap(),
                eq_mle.unwrap().as_extension_packed().unwrap(),
                missing_mul_factor.unwrap_or(EF::ONE),
                sum,
                |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
            ),
            _ => unimplemented!(),
        };
        return vec![
            (PF::<EF>::ZERO, poly.coeffs[0]),
            (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
        ];
    }

    match group {
        MleGroupRef::ExtensionPacked(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            sumcheck_compute_packed::<EF, EFPacking<EF>, _>(
                multilinears,
                zs,
                skips,
                eq_mle,
                folding_factors,
                computation_packed,
                batching_scalars,
                missing_mul_factor,
                packed_fold_size,
            )
        }
        MleGroupRef::BasePacked(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            sumcheck_compute_packed::<EF, PFPacking<EF>, _>(
                multilinears,
                zs,
                skips,
                eq_mle,
                folding_factors,
                computation_packed,
                batching_scalars,
                missing_mul_factor,
                packed_fold_size,
            )
        }
        MleGroupRef::Base(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
            sumcheck_compute_not_packed(
                multilinears,
                zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                batching_scalars,
                missing_mul_factor,
                fold_size,
            )
        }
        MleGroupRef::Extension(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
            sumcheck_compute_not_packed(
                multilinears,
                zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                batching_scalars,
                missing_mul_factor,
                fold_size,
            )
        }
    }
}

pub fn fold_and_sumcheck_compute<'a, EF: ExtensionField<PF<EF>>, SC, SCP>(
    prev_folding_factors: &[EF],
    group: &MleGroupRef<'a, EF>,
    params: SumcheckComputeParams<'a, EF, SC, SCP>,
    zs: &[usize],
) -> (Vec<(PF<EF>, EF)>, MleGroupOwned<EF>)
where
    SC: SumcheckComputation<PF<EF>, EF> + SumcheckComputation<EF, EF> + 'static,
    SCP: SumcheckComputationPacked<EF>,
{
    let SumcheckComputeParams {
        skips,
        eq_mle,
        folding_factors,
        computation,
        computation_packed,
        first_eq_factor,
        batching_scalars,
        missing_mul_factor,
        sum,
    } = params;

    let fold_size = 1 << (group.n_vars() - skips);
    let packed_fold_size = if group.is_packed() {
        fold_size / packing_width::<EF>()
    } else {
        fold_size
    };

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<ProductComputation>() && eq_mle.is_none() {
        assert!(missing_mul_factor.is_none());
        assert!(batching_scalars.is_empty());
        assert_eq!(group.n_columns(), 2);
        assert!(
            prev_folding_factors.len() == 2
                && prev_folding_factors[0] == EF::ONE - prev_folding_factors[1]
        );
        let alpha = prev_folding_factors[1];

        let (poly, folded) = match group {
            MleGroupRef::Extension(multilinears) => {
                let (poly, folded) = fold_and_compute_product_sumcheck_polynomial(
                    &multilinears[0],
                    &multilinears[1],
                    alpha,
                    sum,
                    |e| vec![e],
                );
                (poly, MleGroupOwned::Extension(folded.to_vec()))
            }
            MleGroupRef::ExtensionPacked(multilinears) => {
                let (poly, folded) = fold_and_compute_product_sumcheck_polynomial(
                    &multilinears[0],
                    &multilinears[1],
                    alpha,
                    sum,
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                (poly, MleGroupOwned::ExtensionPacked(folded.to_vec()))
            }
            _ => unimplemented!(),
        };
        return (
            vec![
                (PF::<EF>::ZERO, poly.coeffs[0]),
                (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
            ],
            folded,
        );
    }

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<EF>>() {
        assert!(eq_mle.is_some());
        assert!(batching_scalars.is_empty());
        assert_eq!(group.n_columns(), 4);

        let sc_computation =
            unsafe { std::mem::transmute::<&SC, &GKRQuotientComputation<EF>>(computation) };

        let (poly, folded_multilinears) = match group {
            MleGroupRef::Extension(multilinears) => {
                let (poly, folded) = fold_and_compute_gkr_quotient_sumcheck_polynomial(
                    &prev_folding_factors,
                    &multilinears[0],
                    &multilinears[1],
                    &multilinears[2],
                    &multilinears[3],
                    sc_computation.u4_const,
                    sc_computation.u5_const,
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sum,
                    |e| vec![e],
                );
                let folded = MleGroupOwned::Extension(folded.to_vec());
                (poly, folded)
            }
            MleGroupRef::ExtensionPacked(multilinears) => {
                let (poly, folded) = fold_and_compute_gkr_quotient_sumcheck_polynomial(
                    &prev_folding_factors,
                    &multilinears[0],
                    &multilinears[1],
                    &multilinears[2],
                    &multilinears[3],
                    sc_computation.u4_const,
                    sc_computation.u5_const,
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension_packed().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sum,
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                let folded = MleGroupOwned::ExtensionPacked(folded.to_vec());
                (poly, folded)
            }
            _ => unimplemented!(),
        };
        return (
            vec![
                (PF::<EF>::ZERO, poly.coeffs[0]),
                (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
            ],
            folded_multilinears,
        );
    }

    todo!()

    // match group {
    //     MleGroupRef::ExtensionPacked(multilinears) => {
    //         let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
    //         sumcheck_compute_packed::<EF, EFPacking<EF>, _>(
    //             multilinears,
    //             zs,
    //             skips,
    //             eq_mle,
    //             folding_factors,
    //             computation_packed,
    //             batching_scalars,
    //             missing_mul_factor,
    //             packed_fold_size,
    //         )
    //     }
    //     MleGroupRef::BasePacked(multilinears) => {
    //         let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
    //         sumcheck_compute_packed::<EF, PFPacking<EF>, _>(
    //             multilinears,
    //             zs,
    //             skips,
    //             eq_mle,
    //             folding_factors,
    //             computation_packed,
    //             batching_scalars,
    //             missing_mul_factor,
    //             packed_fold_size,
    //         )
    //     }
    //     MleGroupRef::Base(multilinears) => {
    //         let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
    //         sumcheck_compute_not_packed(
    //             multilinears,
    //             zs,
    //             skips,
    //             eq_mle,
    //             folding_factors,
    //             computation,
    //             batching_scalars,
    //             missing_mul_factor,
    //             fold_size,
    //         )
    //     }
    //     MleGroupRef::Extension(multilinears) => {
    //         let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
    //         sumcheck_compute_not_packed(
    //             multilinears,
    //             zs,
    //             skips,
    //             eq_mle,
    //             folding_factors,
    //             computation,
    //             batching_scalars,
    //             missing_mul_factor,
    //             fold_size,
    //         )
    //     }
    // }
}

#[derive(Debug)]
pub struct SumcheckComputeParams<'a, EF: ExtensionField<PF<EF>>, SC, SCP> {
    pub skips: usize,
    pub eq_mle: Option<&'a MleOwned<EF>>,
    pub first_eq_factor: Option<EF>,
    pub folding_factors: &'a [Vec<PF<EF>>],
    pub computation: &'a SC,
    pub computation_packed: &'a SCP,
    pub batching_scalars: &'a [EF],
    pub missing_mul_factor: Option<EF>,
    pub sum: EF,
}

fn sumcheck_compute_not_packed<
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>>,
    SC,
>(
    multilinears: &[&[IF]],
    zs: &[usize],
    skips: usize,
    eq_mle: Option<&[EF]>,
    folding_factors: &[Vec<PF<EF>>],
    computation: &SC,
    batching_scalars: &[EF],
    missing_mul_factor: Option<EF>,
    fold_size: usize,
) -> Vec<(PF<EF>, EF)>
where
    SC: SumcheckComputation<IF, EF>,
{
    let n = zs.len();
    let sum_zs_packed = (0..fold_size)
        .into_par_iter()
        .map(|i| {
            let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
            let rows = multilinears
                .iter()
                .map(|m| {
                    (0..1 << skips)
                        .map(|j| m[i + j * fold_size])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (0..n)
                .map(|z_index| {
                    let folding_factors_z = &folding_factors[z_index];
                    let point = rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(folding_factors_z.iter())
                                .map(|(x, s)| *x * *s)
                                .sum::<IF>()
                        })
                        .collect::<Vec<_>>();

                    let mut res = computation.eval(&point, batching_scalars);
                    if let Some(eq_mle_eval) = eq_mle_eval {
                        res *= eq_mle_eval;
                    }
                    res
                })
                .collect::<Vec<_>>()
        })
        .reduce(
            || vec![EF::ZERO; n],
            |mut acc, sums| {
                sums.into_iter().enumerate().for_each(|(i, sum)| {
                    acc[i] += sum;
                });
                acc
            },
        );
    let mut evals = vec![];
    for (z_index, z) in zs.iter().enumerate() {
        let mut sum_z = sum_zs_packed[z_index];
        if let Some(missing_mul_factor) = missing_mul_factor {
            sum_z *= missing_mul_factor;
        }
        evals.push((PF::<EF>::from_usize(*z), sum_z));
    }
    evals
}

fn sumcheck_compute_packed<
    EF: ExtensionField<PF<EF>>, // extension field
    WPF: PrimeCharacteristicRing + Mul<PFPacking<EF>, Output = WPF> + Copy + Send + Sync + 'static, // witness packed field (either base or extension)
    SCP: SumcheckComputationPacked<EF>,
>(
    multilinears: &[&[WPF]],
    zs: &[usize],
    skips: usize,
    eq_mle: Option<&[EFPacking<EF>]>,
    folding_factors: &[Vec<PF<EF>>],
    computation_packed: &SCP,
    batching_scalars: &[EF],
    missing_mul_factor: Option<EF>,
    packed_fold_size: usize,
) -> Vec<(PF<EF>, EF)> {
    let n = zs.len();
    let folding_factors = folding_factors
        .iter()
        .map(|scalars| {
            scalars
                .iter()
                .map(|s| PFPacking::<EF>::from(*s))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let sum_zs_packed = (0..packed_fold_size)
        .into_par_iter()
        .map(|i| {
            let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
            let rows = multilinears
                .iter()
                .map(|m| {
                    (0..1 << skips)
                        .map(|j| m[i + j * packed_fold_size])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (0..n)
                .map(|z_index| {
                    let folding_factors_z = &folding_factors[z_index];
                    let point = rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(folding_factors_z.iter())
                                .map(|(x, s)| *x * *s)
                                .sum::<WPF>()
                        })
                        .collect::<Vec<_>>();
                    let mut res = if TypeId::of::<WPF>() == TypeId::of::<PFPacking<EF>>() {
                        let point =
                            unsafe { std::mem::transmute::<Vec<WPF>, Vec<PFPacking<EF>>>(point) };
                        computation_packed.eval_packed_base(&point, batching_scalars)
                    } else {
                        let point =
                            unsafe { std::mem::transmute::<Vec<WPF>, Vec<EFPacking<EF>>>(point) };
                        computation_packed.eval_packed_extension(&point, batching_scalars)
                    };
                    if let Some(eq_mle_eval) = eq_mle_eval {
                        res *= eq_mle_eval;
                    }
                    res
                })
                .collect::<Vec<_>>()
        })
        .reduce(
            || vec![EFPacking::<EF>::ZERO; n],
            |mut acc, sums| {
                sums.into_iter().enumerate().for_each(|(i, sum)| {
                    acc[i] += sum;
                });
                acc
            },
        );

    let mut evals = vec![];
    for (z_index, z) in zs.iter().enumerate() {
        let mut sum_z = EFPacking::<EF>::to_ext_iter([sum_zs_packed[z_index]]).sum::<EF>();
        if let Some(missing_mul_factor) = missing_mul_factor {
            sum_z *= missing_mul_factor;
        }
        evals.push((PF::<EF>::from_usize(*z), sum_z));
    }
    evals
}
