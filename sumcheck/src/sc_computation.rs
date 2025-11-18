use crate::*;
use backend::*;
use constraints_folder::*;
use fiat_shamir::*;
use p3_air::Air;
use p3_field::ExtensionField;
use p3_field::PackedFieldExtension;
use p3_field::PrimeCharacteristicRing;
use p3_field::dot_product;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::any::TypeId;
use std::ops::Add;
use std::ops::Mul;

pub trait SumcheckComputation<EF: ExtensionField<PF<EF>>>: Sync {
    fn degree(&self) -> usize;
    fn eval_base(&self, point: &[PF<EF>], alpha_powers: &[EF]) -> EF;
    fn eval_extension(&self, point: &[EF], alpha_powers: &[EF]) -> EF;
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF>;
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF>;
}

pub trait SumcheckComputationForAir {}

impl<EF, A> SumcheckComputation<EF> for A
where
    EF: ExtensionField<PF<EF>>,
    A: SumcheckComputationForAir
        + Send
        + Sync
        + for<'a> Air<ConstraintFolderPackedBase<'a, EF>>
        + for<'a> Air<ConstraintFolderPackedExtension<'a, EF>>
        + for<'a> Air<ConstraintFolder<'a, PF<EF>, EF>>
        + for<'a> Air<ConstraintFolder<'a, EF, EF>>,
{
    #[inline(always)]
    fn eval_base(&self, point: &[PF<EF>], alpha_powers: &[EF]) -> EF {
        let mut folder = ConstraintFolder {
            main: point,
            alpha_powers,
            accumulator: EF::ZERO,
            constraint_index: 0,
        };
        Air::<ConstraintFolder<PF<EF>, EF>>::eval(self, &mut folder);
        folder.accumulator
    }

    #[inline(always)]
    fn eval_extension(&self, point: &[EF], alpha_powers: &[EF]) -> EF {
        let mut folder = ConstraintFolder {
            main: point,
            alpha_powers,
            accumulator: EF::ZERO,
            constraint_index: 0,
        };
        Air::<ConstraintFolder<EF, EF>>::eval(self, &mut folder);
        folder.accumulator
    }

    #[inline(always)]
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF> {
        let mut folder = ConstraintFolderPackedBase {
            main: point,
            alpha_powers,
            accumulator: Default::default(),
            constraint_index: 0,
        };
        Air::<ConstraintFolderPackedBase<_>>::eval(self, &mut folder);

        folder.accumulator
    }

    #[inline(always)]
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF> {
        let mut folder = ConstraintFolderPackedExtension {
            main: point,
            alpha_powers,
            accumulator: Default::default(),
            constraint_index: 0,
        };
        Air::<ConstraintFolderPackedExtension<_>>::eval(self, &mut folder);

        folder.accumulator
    }

    fn degree(&self) -> usize {
        <A as Air<ConstraintFolder<EF, EF>>>::degree(self)
    }
}

pub fn sumcheck_compute<'a, EF: ExtensionField<PF<EF>>, SC>(
    group: &MleGroupRef<'a, EF>,
    params: SumcheckComputeParams<'a, EF, SC>,
    zs: &[usize],
) -> Vec<(PF<EF>, EF)>
where
    SC: SumcheckComputation<EF> + 'static,
{
    let SumcheckComputeParams {
        skips,
        eq_mle,
        first_eq_factor,
        folding_factors,
        computation,
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
    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<2>>() {
        assert!(eq_mle.is_some());
        assert_eq!(group.n_columns(), 4);

        let poly = match group {
            MleGroupRef::Extension(multilinears) => compute_gkr_quotient_sumcheck_polynomial(
                &multilinears[0],
                &multilinears[1],
                &multilinears[2],
                &multilinears[3],
                batching_scalars[1],
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
                batching_scalars[1],
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
                computation,
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
                computation,
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

pub fn fold_and_sumcheck_compute<'a, EF: ExtensionField<PF<EF>>, SC>(
    prev_folding_factors: &[EF],
    group: &MleGroupRef<'a, EF>,
    params: SumcheckComputeParams<'a, EF, SC>,
    zs: &[usize],
) -> (Vec<(PF<EF>, EF)>, MleGroupOwned<EF>)
where
    SC: SumcheckComputation<EF> + 'static,
{
    let SumcheckComputeParams {
        skips,
        eq_mle,
        first_eq_factor,
        folding_factors,
        computation,
        batching_scalars,
        missing_mul_factor,
        sum,
    } = params;

    let fold_size = 1 << (group.n_vars() - skips - log2_strict_usize(prev_folding_factors.len()));
    let compute_fold_size = if group.is_packed() {
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
        let prev_folding_factor = prev_folding_factors[1];

        let (poly, folded) = match group {
            MleGroupRef::Extension(multilinears) => {
                let (poly, folded) = fold_and_compute_product_sumcheck_polynomial(
                    &multilinears[0],
                    &multilinears[1],
                    prev_folding_factor,
                    sum,
                    |e| vec![e],
                );
                (poly, MleGroupOwned::Extension(folded))
            }
            MleGroupRef::ExtensionPacked(multilinears) => {
                let (poly, folded) = fold_and_compute_product_sumcheck_polynomial(
                    &multilinears[0],
                    &multilinears[1],
                    prev_folding_factor,
                    sum,
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                (poly, MleGroupOwned::ExtensionPacked(folded))
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
    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<2>>() {
        assert!(eq_mle.is_some());
        assert_eq!(group.n_columns(), 4);
        assert!(
            prev_folding_factors.len() == 2
                && prev_folding_factors[0] == EF::ONE - prev_folding_factors[1]
        );
        let prev_folding_factor = prev_folding_factors[1];

        let (poly, folded_multilinears) = match group {
            MleGroupRef::Extension(multilinears) => {
                let (poly, folded) = fold_and_compute_gkr_quotient_sumcheck_polynomial(
                    prev_folding_factor,
                    &multilinears[0],
                    &multilinears[1],
                    &multilinears[2],
                    &multilinears[3],
                    batching_scalars[1],
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sum,
                    |e| vec![e],
                );
                let folded = MleGroupOwned::Extension(folded);
                (poly, folded)
            }
            MleGroupRef::ExtensionPacked(multilinears) => {
                let (poly, folded) = fold_and_compute_gkr_quotient_sumcheck_polynomial(
                    prev_folding_factor,
                    &multilinears[0],
                    &multilinears[1],
                    &multilinears[2],
                    &multilinears[3],
                    batching_scalars[1],
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension_packed().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sum,
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                let folded = MleGroupOwned::ExtensionPacked(folded);
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

    match group {
        MleGroupRef::ExtensionPacked(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            sumcheck_fold_and_compute_packed::<EF, EFPacking<EF>, _>(
                prev_folding_factors,
                multilinears,
                zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                batching_scalars,
                missing_mul_factor,
                compute_fold_size,
                |wpf, ef| wpf * ef,
            )
        }
        MleGroupRef::BasePacked(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            sumcheck_fold_and_compute_packed::<EF, PFPacking<EF>, _>(
                prev_folding_factors,
                multilinears,
                zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                batching_scalars,
                missing_mul_factor,
                compute_fold_size,
                |wpf, ef| EFPacking::<EF>::from(ef) * wpf,
            )
        }
        MleGroupRef::Base(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
            sumcheck_fold_and_compute_not_packed::<EF, PF<EF>, _>(
                prev_folding_factors,
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
            sumcheck_fold_and_compute_not_packed::<EF, EF, _>(
                prev_folding_factors,
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

#[derive(Debug)]
pub struct SumcheckComputeParams<'a, EF: ExtensionField<PF<EF>>, SC> {
    pub skips: usize,
    pub eq_mle: Option<&'a MleOwned<EF>>,
    pub first_eq_factor: Option<EF>,
    pub folding_factors: &'a [Vec<PF<EF>>],
    pub computation: &'a SC,
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
    SC: SumcheckComputation<EF>,
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

                    let mut res = if TypeId::of::<IF>() == TypeId::of::<PF<EF>>() {
                        let point = unsafe { std::mem::transmute::<Vec<IF>, Vec<PF<EF>>>(point) };
                        computation.eval_base(&point, batching_scalars)
                    } else {
                        assert!(TypeId::of::<IF>() == TypeId::of::<EF>());
                        let point = unsafe { std::mem::transmute::<Vec<IF>, Vec<EF>>(point) };
                        computation.eval_extension(&point, batching_scalars)
                    };
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

fn sumcheck_fold_and_compute_not_packed<
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>>,
    SC,
>(
    prev_folding_factors: &[EF],
    multilinears: &[&[IF]],
    zs: &[usize],
    skips: usize,
    eq_mle: Option<&[EF]>,
    folding_factors: &[Vec<PF<EF>>],
    computation: &SC,
    batching_scalars: &[EF],
    missing_mul_factor: Option<EF>,
    compute_fold_size: usize,
) -> (Vec<(PF<EF>, EF)>, MleGroupOwned<EF>)
where
    SC: SumcheckComputation<EF>,
{
    let bi_folded = prev_folding_factors.len() == 2;
    if bi_folded {
        assert!(prev_folding_factors[0] == EF::ONE - prev_folding_factors[1],);
    }
    let prev_folded_size = multilinears[0].len() / prev_folding_factors.len();
    let folded = (0..multilinears.len())
        .map(|_| EF::zero_vec(prev_folded_size))
        .collect::<Vec<_>>();
    let n = zs.len();
    let sum_zs_packed = (0..compute_fold_size)
        .into_par_iter()
        .map(|i| {
            let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
            let rows = multilinears
                .iter()
                .enumerate()
                .map(|(j, m)| {
                    (0..1 << skips)
                        .map(|k| {
                            let id = i + k * compute_fold_size;
                            let res: EF = if bi_folded {
                                prev_folding_factors[1] * (m[id + prev_folded_size] - m[id]) + m[id]
                            } else {
                                dot_product(
                                    prev_folding_factors.iter().copied(),
                                    (0..prev_folding_factors.len())
                                        .map(|l| m[id + l * prev_folded_size]),
                                )
                            };

                            unsafe {
                                // folded[j][id] = res;
                                let folded_ptr = folded[j].as_ptr() as *mut EF;
                                *folded_ptr.add(id) = res;
                            }
                            res
                        })
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
                                .sum::<EF>()
                        })
                        .collect::<Vec<_>>();

                    let mut res = computation.eval_extension(&point, batching_scalars);
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
    (evals, MleGroupOwned::Extension(folded))
}

fn sumcheck_compute_packed<
    EF: ExtensionField<PF<EF>>, // extension field
    WPF: PrimeCharacteristicRing + Mul<PFPacking<EF>, Output = WPF> + Copy + Send + Sync + 'static, // witness packed field (either base or extension)
    SCP: SumcheckComputation<EF>,
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

fn sumcheck_fold_and_compute_packed<
    EF: ExtensionField<PF<EF>>, // extension field
    WPF: PrimeCharacteristicRing + Mul<PFPacking<EF>, Output = WPF> + Copy + Send + Sync + 'static, // witness packed field (either base or extension)
    SCP: SumcheckComputation<EF>,
>(
    prev_folding_factors: &[EF],
    multilinears: &[&[WPF]],
    zs: &[usize],
    skips: usize,
    eq_mle: Option<&[EFPacking<EF>]>,
    folding_factors: &[Vec<PF<EF>>],
    computation_packed: &SCP,
    batching_scalars: &[EF],
    missing_mul_factor: Option<EF>,
    compute_fold_size: usize,
    mul: impl Fn(WPF, EF) -> EFPacking<EF> + Sync + Send,
) -> (Vec<(PF<EF>, EF)>, MleGroupOwned<EF>)
where
    EFPacking<EF>: Add<WPF, Output = EFPacking<EF>>,
{
    let prev_folded_size = multilinears[0].len() / prev_folding_factors.len();
    let folded = (0..multilinears.len())
        .map(|_| EFPacking::<EF>::zero_vec(prev_folded_size))
        .collect::<Vec<_>>();
    let n = zs.len();

    let bi_folded = prev_folding_factors.len() == 2;
    if bi_folded {
        assert!(prev_folding_factors[0] == EF::ONE - prev_folding_factors[1],);
    }

    let folding_factors = folding_factors
        .iter()
        .map(|scalars| {
            scalars
                .iter()
                .map(|s| PFPacking::<EF>::from(*s))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let sum_zs_packed = (0..compute_fold_size)
        .into_par_iter()
        .map(|i| {
            let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
            let rows = multilinears
                .iter()
                .enumerate()
                .map(|(j, m)| {
                    (0..1 << skips)
                        .map(|k| {
                            let id = i + k * compute_fold_size;
                            let res: EFPacking<EF> = if bi_folded {
                                mul(m[id + prev_folded_size] - m[id], prev_folding_factors[1])
                                    + m[id]
                            } else {
                                prev_folding_factors
                                    .iter()
                                    .enumerate()
                                    .map(|(l, &f)| mul(m[id + l * prev_folded_size], f))
                                    .sum()
                            };
                            unsafe {
                                // folded[j][id] = res;
                                let folded_ptr = folded[j].as_ptr() as *mut EFPacking<EF>;
                                *folded_ptr.add(id) = res;
                            }
                            res
                        })
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
                                .sum::<EFPacking<EF>>()
                        })
                        .collect::<Vec<_>>();
                    let mut res =
                        computation_packed.eval_packed_extension(&point, batching_scalars);
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
    (evals, MleGroupOwned::ExtensionPacked(folded))
}
