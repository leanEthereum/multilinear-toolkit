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
use unroll_macro::unroll_match;

pub trait SumcheckComputation<EF: ExtensionField<PF<EF>>>: Sync + 'static {
    type ExtraData: Send + Sync + 'static;
    const N_STEPS: usize = 1;

    fn degrees(&self) -> Vec<usize>; // should be in increasing order

    fn max_degree(&self) -> usize {
        self.degrees().into_iter().max().unwrap()
    }

    fn eval_base<const STEP: usize>(
        &self,
        point_f: &[PF<EF>],
        point_ef: &[EF],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EF;
    fn eval_extension<const STEP: usize>(
        &self,
        point_f: &[EF],
        point_ef: &[EF],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EF;
    fn eval_packed_base<const STEP: usize>(
        &self,
        point_f: &[PFPacking<EF>],
        point_ef: &[EFPacking<EF>],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EFPacking<EF>;
    fn eval_packed_extension<const STEP: usize>(
        &self,
        point_f: &[EFPacking<EF>],
        point_ef: &[EFPacking<EF>],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EFPacking<EF>;

    #[inline(always)]
    fn n_steps(&self) -> usize {
        self.degrees().len()
    }

    fn eval_extension_everywhere(
        &self,
        point_f: &[EF],
        point_ef: &[EF],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EF {
        let mut res = EF::ZERO;

        unroll_match!(Self::N_STEPS, I, {
            res += self.eval_extension::<I>(point_f, point_ef, extra_data, alpha_powers);
        });

        res
    }
}

impl<EF, A> SumcheckComputation<EF> for A
where
    EF: ExtensionField<PF<EF>>,
    A: Send + Sync + Air,
{
    type ExtraData = A::ExtraData;

    #[inline(always)]
    fn eval_base<const STEP: usize>(
        &self,
        point_f: &[PF<EF>],
        point_ef: &[EF],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EF {
        let mut folder = ConstraintFolder {
            up_f: &point_f[..self.n_columns_f_air()],
            down_f: &point_f[self.n_columns_f_air()..],
            up_ef: &point_ef[..self.n_columns_ef_air()],
            down_ef: &point_ef[self.n_columns_ef_air()..],
            extra_data,
            alpha_powers: &alpha_powers[self.n_constraints_before_step(STEP)..],
            accumulator: EF::ZERO,
            constraint_index: 0,
        };
        Air::eval::<_, STEP>(self, &mut folder, extra_data);
        folder.accumulator
    }

    #[inline(always)]
    fn eval_extension<const STEP: usize>(
        &self,
        point_f: &[EF],
        point_ef: &[EF],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EF {
        let mut folder = ConstraintFolder {
            up_f: &point_f[..self.n_columns_f_air()],
            down_f: &point_f[self.n_columns_f_air()..],
            up_ef: &point_ef[..self.n_columns_ef_air()],
            down_ef: &point_ef[self.n_columns_ef_air()..],
            extra_data,
            alpha_powers: &alpha_powers[self.n_constraints_before_step(STEP)..],
            accumulator: EF::ZERO,
            constraint_index: 0,
        };
        Air::eval::<_, STEP>(self, &mut folder, extra_data);
        folder.accumulator
    }

    #[inline(always)]
    fn eval_packed_base<const STEP: usize>(
        &self,
        point_f: &[PFPacking<EF>],
        point_ef: &[EFPacking<EF>],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EFPacking<EF> {
        let mut folder = ConstraintFolderPackedBase {
            up_f: &point_f[..self.n_columns_f_air()],
            down_f: &point_f[self.n_columns_f_air()..],
            up_ef: &point_ef[..self.n_columns_ef_air()],
            down_ef: &point_ef[self.n_columns_ef_air()..],
            extra_data,
            alpha_powers: &alpha_powers[self.n_constraints_before_step(STEP)..],
            accumulator: Default::default(),
            constraint_index: 0,
        };
        Air::eval::<_, STEP>(self, &mut folder, extra_data);
        folder.accumulator
    }

    #[inline(always)]
    fn eval_packed_extension<const STEP: usize>(
        &self,
        point_f: &[EFPacking<EF>],
        point_ef: &[EFPacking<EF>],
        extra_data: &Self::ExtraData,
        alpha_powers: &[EF],
    ) -> EFPacking<EF> {
        let mut folder = ConstraintFolderPackedExtension {
            up_f: &point_f[..self.n_columns_f_air()],
            down_f: &point_f[self.n_columns_f_air()..],
            up_ef: &point_ef[..self.n_columns_ef_air()],
            down_ef: &point_ef[self.n_columns_ef_air()..],
            extra_data,
            alpha_powers: &alpha_powers[self.n_constraints_before_step(STEP)..],
            accumulator: Default::default(),
            constraint_index: 0,
        };
        Air::eval::<_, STEP>(self, &mut folder, extra_data);
        folder.accumulator
    }

    fn degrees(&self) -> Vec<usize> {
        self.degrees()
    }
}

pub fn sumcheck_compute<'a, EF: ExtensionField<PF<EF>>, SC>(
    group_f: &MleGroupRef<'a, EF>,
    group_ef: &MleGroupRef<'a, EF>,
    params: SumcheckComputeParams<'a, EF, SC>,
    all_zs: &Vec<Vec<usize>>,
) -> Vec<Vec<(PF<EF>, EF)>>
where
    SC: SumcheckComputation<EF> + 'static,
{
    let SumcheckComputeParams {
        skips,
        eq_mle,
        first_eq_factor,
        folding_factors,
        computation,
        extra_data,
        alpha_powers,
        missing_mul_factor,
        sums,
    } = params;

    let fold_size = 1 << (group_f.n_vars() - skips);
    let packed_fold_size = if group_f.is_packed() {
        fold_size / packing_width::<EF>()
    } else {
        fold_size
    };

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<ProductComputation>() && eq_mle.is_none() {
        assert!(missing_mul_factor.is_none());
        assert!(alpha_powers.is_empty());
        assert_eq!(group_f.n_columns(), 2);
        assert_eq!(group_ef.n_columns(), 0);
        assert_eq!(sums.len(), 1);

        let poly = match group_f {
            MleGroupRef::Extension(multilinears) => compute_product_sumcheck_polynomial(
                &multilinears[0],
                &multilinears[1],
                sums[0],
                |e| vec![e],
            ),
            MleGroupRef::ExtensionPacked(multilinears) => compute_product_sumcheck_polynomial(
                &multilinears[0],
                &multilinears[1],
                sums[0],
                |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
            ),
            _ => unimplemented!(),
        };
        return vec![vec![
            (PF::<EF>::ZERO, poly.coeffs[0]),
            (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
        ]];
    }

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<2>>() {
        assert!(eq_mle.is_some());
        assert_eq!(group_f.n_columns(), 4);
        assert_eq!(group_ef.n_columns(), 0);
        assert_eq!(sums.len(), 1);

        let poly = match group_f {
            MleGroupRef::Extension(multilinears) => compute_gkr_quotient_sumcheck_polynomial(
                &multilinears[0],
                &multilinears[1],
                &multilinears[2],
                &multilinears[3],
                alpha_powers[1],
                first_eq_factor.unwrap(),
                eq_mle.unwrap().as_extension().unwrap(),
                missing_mul_factor.unwrap_or(EF::ONE),
                sums[0],
                |e| vec![e],
            ),
            MleGroupRef::ExtensionPacked(multilinears) => compute_gkr_quotient_sumcheck_polynomial(
                &multilinears[0],
                &multilinears[1],
                &multilinears[2],
                &multilinears[3],
                alpha_powers[1],
                first_eq_factor.unwrap(),
                eq_mle.unwrap().as_extension_packed().unwrap(),
                missing_mul_factor.unwrap_or(EF::ONE),
                sums[0],
                |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
            ),
            _ => unimplemented!(),
        };
        return vec![vec![
            (PF::<EF>::ZERO, poly.coeffs[0]),
            (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
        ]];
    }

    match group_f {
        MleGroupRef::ExtensionPacked(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            let multilinears_ef = group_ef.as_extension_packed().unwrap();
            sumcheck_compute_packed::<EF, EFPacking<EF>, _>(
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                packed_fold_size,
            )
        }
        MleGroupRef::BasePacked(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            let multilinears_ef = group_ef.as_extension_packed().unwrap();
            sumcheck_compute_packed::<EF, PFPacking<EF>, _>(
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                packed_fold_size,
            )
        }
        MleGroupRef::Base(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
            let multilinears_ef = group_ef.as_extension().unwrap();
            sumcheck_compute_not_packed(
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                fold_size,
            )
        }
        MleGroupRef::Extension(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
            let multilinears_ef = group_ef.as_extension().unwrap();
            sumcheck_compute_not_packed(
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                fold_size,
            )
        }
    }
}

pub fn fold_and_sumcheck_compute<'a, EF: ExtensionField<PF<EF>>, SC: SumcheckComputation<EF>>(
    prev_folding_factors: &[EF],
    group_f: &MleGroupRef<'a, EF>,
    group_ef: &MleGroupRef<'a, EF>,
    params: SumcheckComputeParams<'a, EF, SC>,
    all_zs: &[Vec<usize>],
) -> (Vec<Vec<(PF<EF>, EF)>>, MleGroupOwned<EF>, MleGroupOwned<EF>) {
    let SumcheckComputeParams {
        skips,
        eq_mle,
        first_eq_factor,
        folding_factors,
        computation,
        extra_data,
        alpha_powers,
        missing_mul_factor,
        sums,
    } = params;

    let fold_size = 1 << (group_f.n_vars() - skips - log2_strict_usize(prev_folding_factors.len()));
    let compute_fold_size = if group_f.is_packed() {
        fold_size / packing_width::<EF>()
    } else {
        fold_size
    };

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<ProductComputation>() && eq_mle.is_none() {
        assert!(missing_mul_factor.is_none());
        assert!(alpha_powers.is_empty());
        assert_eq!(group_f.n_columns(), 2);
        assert_eq!(group_ef.n_columns(), 0);
        assert!(
            prev_folding_factors.len() == 2
                && prev_folding_factors[0] == EF::ONE - prev_folding_factors[1]
        );
        assert_eq!(sums.len(), 1);

        let prev_folding_factor = prev_folding_factors[1];

        let (poly, folded_f) = match group_f {
            MleGroupRef::Extension(multilinears) => {
                let (poly, folded) = fold_and_compute_product_sumcheck_polynomial(
                    &multilinears[0],
                    &multilinears[1],
                    prev_folding_factor,
                    sums[0],
                    |e| vec![e],
                );
                (poly, MleGroupOwned::Extension(folded))
            }
            MleGroupRef::ExtensionPacked(multilinears) => {
                let (poly, folded) = fold_and_compute_product_sumcheck_polynomial(
                    &multilinears[0],
                    &multilinears[1],
                    prev_folding_factor,
                    sums[0],
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                (poly, MleGroupOwned::ExtensionPacked(folded))
            }
            _ => unimplemented!(),
        };
        let folded_ef = MleGroupOwned::empty(true, folded_f.is_packed());
        return (
            vec![vec![
                (PF::<EF>::ZERO, poly.coeffs[0]),
                (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
            ]],
            folded_f,
            folded_ef,
        );
    }

    // TODO handle this in a more general way
    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<2>>() {
        assert!(eq_mle.is_some());
        assert_eq!(group_f.n_columns(), 4);
        assert_eq!(group_ef.n_columns(), 0);
        assert!(
            prev_folding_factors.len() == 2
                && prev_folding_factors[0] == EF::ONE - prev_folding_factors[1]
        );
        assert_eq!(sums.len(), 1);

        let prev_folding_factor = prev_folding_factors[1];

        let (poly, folded_f) = match group_f {
            MleGroupRef::Extension(multilinears) => {
                let (poly, folded) = fold_and_compute_gkr_quotient_sumcheck_polynomial(
                    prev_folding_factor,
                    &multilinears[0],
                    &multilinears[1],
                    &multilinears[2],
                    &multilinears[3],
                    alpha_powers[1],
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sums[0],
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
                    alpha_powers[1],
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension_packed().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sums[0],
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                let folded = MleGroupOwned::ExtensionPacked(folded);
                (poly, folded)
            }
            _ => unimplemented!(),
        };
        let folded_ef = MleGroupOwned::empty(true, folded_f.is_packed());
        return (
            vec![vec![
                (PF::<EF>::ZERO, poly.coeffs[0]),
                (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
            ]],
            folded_f,
            folded_ef,
        );
    }

    match group_f {
        MleGroupRef::ExtensionPacked(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            let multilinears_ef = group_ef.as_extension_packed().unwrap();
            sumcheck_fold_and_compute_packed::<EF, EFPacking<EF>, _>(
                prev_folding_factors,
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                compute_fold_size,
                |wpf, ef| wpf * ef,
            )
        }
        MleGroupRef::BasePacked(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            let multilinears_ef = group_ef.as_extension_packed().unwrap();
            sumcheck_fold_and_compute_packed::<EF, PFPacking<EF>, _>(
                prev_folding_factors,
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                compute_fold_size,
                |wpf, ef| EFPacking::<EF>::from(ef) * wpf,
            )
        }
        MleGroupRef::Base(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
            let multilinears_ef = group_ef.as_extension().unwrap();
            sumcheck_fold_and_compute_not_packed::<EF, PF<EF>, _>(
                prev_folding_factors,
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                fold_size,
            )
        }
        MleGroupRef::Extension(multilinears_f) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap());
            let multilinears_ef = group_ef.as_extension().unwrap();
            sumcheck_fold_and_compute_not_packed::<EF, EF, _>(
                prev_folding_factors,
                multilinears_f,
                multilinears_ef,
                all_zs,
                skips,
                eq_mle,
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                fold_size,
            )
        }
    }
}

#[derive(Debug)]
pub struct SumcheckComputeParams<'a, EF: ExtensionField<PF<EF>>, SC: SumcheckComputation<EF>> {
    pub skips: usize,
    pub eq_mle: Option<&'a MleOwned<EF>>,
    pub first_eq_factor: Option<EF>,
    pub folding_factors: &'a [Vec<Vec<PF<EF>>>],
    pub computation: &'a SC,
    pub extra_data: &'a SC::ExtraData,
    pub alpha_powers: &'a [EF],
    pub missing_mul_factor: Option<EF>,
    pub sums: &'a [EF],
}

fn helper_1<
    const STEP: usize,
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF>,
>(
    zs: &Vec<usize>,
    folding_factors: &Vec<Vec<PF<EF>>>,
    rows_f: &Vec<Vec<IF>>,
    rows_ef: &Vec<Vec<EF>>,
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_mle_eval: Option<EF>,
    res: &mut Vec<Vec<EF>>,
) {
    let n = zs.len();
    let evals = (0..n)
        .map(|z_index| {
            let folding_factors_z = &folding_factors[z_index];
            let point_f = rows_f
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<IF>()
                })
                .collect::<Vec<_>>();
            let point_ef = rows_ef
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<EF>()
                })
                .collect::<Vec<_>>();

            let mut res = if TypeId::of::<IF>() == TypeId::of::<PF<EF>>() {
                let point_f = unsafe { std::mem::transmute::<Vec<IF>, Vec<PF<EF>>>(point_f) };
                computation.eval_base::<STEP>(&point_f, &point_ef, extra_data, alpha_powers)
            } else {
                assert!(TypeId::of::<IF>() == TypeId::of::<EF>());
                let point_f = unsafe { std::mem::transmute::<Vec<IF>, Vec<EF>>(point_f) };
                computation.eval_extension::<STEP>(&point_f, &point_ef, extra_data, alpha_powers)
            };
            if let Some(eq_mle_eval) = eq_mle_eval {
                res *= eq_mle_eval;
            }
            res
        })
        .collect::<Vec<_>>();
    res.push(evals);
}

fn sumcheck_compute_not_packed<
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF>,
>(
    multilinears_f: &[&[IF]],
    multilinears_ef: &[&[EF]],
    all_zs: &[Vec<usize>],
    skips: usize,
    eq_mle: Option<&[EF]>,
    all_folding_factors: &[Vec<Vec<PF<EF>>>],
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    missing_mul_factor: Option<EF>,
    fold_size: usize,
) -> Vec<Vec<(PF<EF>, EF)>> {
    let compute_iteration = |i: usize| -> Vec<Vec<EF>> {
        let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
        let rows_f = multilinears_f
            .iter()
            .map(|m| {
                (0..1 << skips)
                    .map(|j| m[i + j * fold_size])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows_ef = multilinears_ef
            .iter()
            .map(|m| {
                (0..1 << skips)
                    .map(|j| m[i + j * fold_size])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut res = Vec::with_capacity(computation.n_steps());
        unroll_match!(SC::N_STEPS, I, {
            helper_1::<I, _, _, _>(
                &all_zs[I],
                &all_folding_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
                &mut res,
            );
        });

        res
    };

    let init_values = all_zs
        .iter()
        .map(|zs| vec![EF::ZERO; zs.len()])
        .collect::<Vec<_>>();
    let sum_values = |current_vals: &mut Vec<Vec<EF>>, new_vals: Vec<Vec<EF>>| {
        current_vals
            .iter_mut()
            .zip(new_vals)
            .for_each(|(curr, new)| {
                curr.iter_mut()
                    .zip(new)
                    .for_each(|(curr, new)| *curr += new);
            });
    };
    let all_sum_zs = if fold_size < PARALLEL_THRESHOLD {
        (0..fold_size).fold(init_values, |mut acc, i| {
            sum_values(&mut acc, compute_iteration(i));
            acc
        })
    } else {
        (0..fold_size)
            .into_par_iter()
            .map(compute_iteration)
            .reduce(
                || init_values.clone(),
                |mut acc, sums| {
                    sum_values(&mut acc, sums);
                    acc
                },
            )
    };

    let mut all_evals = vec![];
    for (zs, sum_zs) in all_zs.iter().zip(all_sum_zs) {
        let mut evals = Vec::with_capacity(zs.len());
        for (z_index, z) in zs.iter().enumerate() {
            let mut sum_z = sum_zs[z_index];
            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= missing_mul_factor;
            }
            evals.push((PF::<EF>::from_usize(*z), sum_z));
        }
        all_evals.push(evals);
    }

    all_evals
}

fn helper_2<const STEP: usize, EF: ExtensionField<PF<EF>>, SC: SumcheckComputation<EF>>(
    zs: &Vec<usize>,
    folding_factors: &Vec<Vec<PF<EF>>>,
    rows_f: &Vec<Vec<EF>>,
    rows_ef: &Vec<Vec<EF>>,
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_mle_eval: Option<EF>,
    res: &mut Vec<Vec<EF>>,
) {
    let n = zs.len();
    let evals = (0..n)
        .map(|z_index| {
            let folding_factors_z = &folding_factors[z_index];
            let point_f = rows_f
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<EF>()
                })
                .collect::<Vec<_>>();
            let point_ef = rows_ef
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<EF>()
                })
                .collect::<Vec<_>>();

            let mut res =
                computation.eval_extension::<STEP>(&point_f, &point_ef, extra_data, alpha_powers);
            if let Some(eq_mle_eval) = eq_mle_eval {
                res *= eq_mle_eval;
            }
            res
        })
        .collect::<Vec<_>>();
    res.push(evals);
}

fn sumcheck_fold_and_compute_not_packed<
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF>,
>(
    prev_folding_factors: &[EF],
    multilinears_f: &[&[IF]],
    multilinears_ef: &[&[EF]],
    all_zs: &[Vec<usize>],
    skips: usize,
    eq_mle: Option<&[EF]>,
    all_folding_factors: &[Vec<Vec<PF<EF>>>],
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    missing_mul_factor: Option<EF>,
    compute_fold_size: usize,
) -> (Vec<Vec<(PF<EF>, EF)>>, MleGroupOwned<EF>, MleGroupOwned<EF>) {
    let bi_folded = prev_folding_factors.len() == 2;
    if bi_folded {
        assert!(prev_folding_factors[0] == EF::ONE - prev_folding_factors[1],);
    }
    let prev_folded_size = multilinears_f[0].len() / prev_folding_factors.len();
    let folded_f = (0..multilinears_f.len())
        .map(|_| EF::zero_vec(prev_folded_size))
        .collect::<Vec<_>>();
    let folded_ef = (0..multilinears_ef.len())
        .map(|_| EF::zero_vec(prev_folded_size))
        .collect::<Vec<_>>();

    let compute_iteration = |i: usize| -> Vec<Vec<EF>> {
        let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
        let rows_f = multilinears_f
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
                            let folded_ptr = folded_f[j].as_ptr() as *mut EF;
                            *folded_ptr.add(id) = res;
                        }
                        res
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows_ef = multilinears_ef
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
                            let folded_ptr = folded_ef[j].as_ptr() as *mut EF;
                            *folded_ptr.add(id) = res;
                        }
                        res
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut res = Vec::with_capacity(computation.n_steps());

        unroll_match!(SC::N_STEPS, I, {
            helper_2::<I, _, _>(
                &all_zs[I],
                &all_folding_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
                &mut res,
            );
        });

        res
    };

    let init_values = all_zs
        .iter()
        .map(|zs| vec![EF::ZERO; zs.len()])
        .collect::<Vec<_>>();
    let sum_values = |current_vals: &mut Vec<Vec<EF>>, new_vals: Vec<Vec<EF>>| {
        current_vals
            .iter_mut()
            .zip(new_vals)
            .for_each(|(curr, new)| {
                curr.iter_mut()
                    .zip(new)
                    .for_each(|(curr, new)| *curr += new);
            });
    };
    let all_sum_zs = if compute_fold_size < PARALLEL_THRESHOLD {
        (0..compute_fold_size).fold(init_values, |mut acc, i| {
            sum_values(&mut acc, compute_iteration(i));
            acc
        })
    } else {
        (0..compute_fold_size)
            .into_par_iter()
            .map(compute_iteration)
            .reduce(
                || init_values.clone(),
                |mut acc, sums| {
                    sum_values(&mut acc, sums);
                    acc
                },
            )
    };

    let mut all_evals = vec![];
    for (zs, sum_zs) in all_zs.iter().zip(all_sum_zs) {
        let mut evals = Vec::with_capacity(zs.len());
        for (z_index, z) in zs.iter().enumerate() {
            let mut sum_z = sum_zs[z_index];
            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= missing_mul_factor;
            }
            evals.push((PF::<EF>::from_usize(*z), sum_z));
        }
        all_evals.push(evals);
    }
    (
        all_evals,
        MleGroupOwned::Extension(folded_f),
        MleGroupOwned::Extension(folded_ef),
    )
}

fn helper_3<
    const STEP: usize,
    EF: ExtensionField<PF<EF>>, // extension field
    WPF: PrimeCharacteristicRing + Mul<PFPacking<EF>, Output = WPF> + Copy + Send + Sync + 'static, // witness packed field (either base or extension)
    SC: SumcheckComputation<EF>,
>(
    zs: &Vec<usize>,
    folding_factors: &Vec<Vec<PFPacking<EF>>>,
    rows_f: &Vec<Vec<WPF>>,
    rows_ef: &Vec<Vec<EFPacking<EF>>>,
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_mle_eval: Option<EFPacking<EF>>,
    res: &mut Vec<Vec<EFPacking<EF>>>,
) {
    let n = zs.len();
    let evals = (0..n)
        .map(|z_index| {
            let folding_factors_z = &folding_factors[z_index];
            let point_f = rows_f
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<WPF>()
                })
                .collect::<Vec<_>>();
            let point_ef = rows_ef
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<EFPacking<EF>>()
                })
                .collect::<Vec<_>>();
            let mut res = if TypeId::of::<WPF>() == TypeId::of::<PFPacking<EF>>() {
                let point_f =
                    unsafe { std::mem::transmute::<Vec<WPF>, Vec<PFPacking<EF>>>(point_f) };
                computation.eval_packed_base::<STEP>(&point_f, &point_ef, extra_data, alpha_powers)
            } else {
                let point_f =
                    unsafe { std::mem::transmute::<Vec<WPF>, Vec<EFPacking<EF>>>(point_f) };
                computation.eval_packed_extension::<STEP>(
                    &point_f,
                    &point_ef,
                    extra_data,
                    alpha_powers,
                )
            };
            if let Some(eq_mle_eval) = eq_mle_eval {
                res *= eq_mle_eval;
            }
            res
        })
        .collect::<Vec<_>>();
    res.push(evals);
}

fn sumcheck_compute_packed<
    EF: ExtensionField<PF<EF>>, // extension field
    WPF: PrimeCharacteristicRing + Mul<PFPacking<EF>, Output = WPF> + Copy + Send + Sync + 'static, // witness packed field (either base or extension)
    SC: SumcheckComputation<EF>,
>(
    multilinears_f: &[&[WPF]],
    multilinears_ef: &[&[EFPacking<EF>]],
    all_zs: &[Vec<usize>],
    skips: usize,
    eq_mle: Option<&[EFPacking<EF>]>,
    all_folding_factors: &[Vec<Vec<PF<EF>>>],
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    missing_mul_factor: Option<EF>,
    packed_fold_size: usize,
) -> Vec<Vec<(PF<EF>, EF)>> {
    let all_folding_factors = all_folding_factors
        .iter()
        .map(|folding_factors| {
            folding_factors
                .iter()
                .map(|scalars| {
                    scalars
                        .iter()
                        .map(|s| PFPacking::<EF>::from(*s))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let compute_iteration = |i: usize| -> Vec<Vec<EFPacking<EF>>> {
        let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
        let rows_f = multilinears_f
            .iter()
            .map(|m| {
                (0..1 << skips)
                    .map(|j| m[i + j * packed_fold_size])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows_ef = multilinears_ef
            .iter()
            .map(|m| {
                (0..1 << skips)
                    .map(|j| m[i + j * packed_fold_size])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut res = Vec::with_capacity(computation.n_steps());
        unroll_match!(SC::N_STEPS, I, {
            helper_3::<I, _, _, _>(
                &all_zs[I],
                &all_folding_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
                &mut res,
            );
        });

        res
    };

    let init_values = all_zs
        .iter()
        .map(|zs| vec![EFPacking::<EF>::ZERO; zs.len()])
        .collect::<Vec<_>>();
    let sum_values = |current_vals: &mut Vec<Vec<EFPacking<EF>>>,
                      new_vals: Vec<Vec<EFPacking<EF>>>| {
        current_vals
            .iter_mut()
            .zip(new_vals)
            .for_each(|(curr, new)| {
                curr.iter_mut()
                    .zip(new)
                    .for_each(|(curr, new)| *curr += new);
            });
    };

    let all_sum_zs_packed = if packed_fold_size < PARALLEL_THRESHOLD {
        (0..packed_fold_size).fold(init_values, |mut acc, i| {
            sum_values(&mut acc, compute_iteration(i));
            acc
        })
    } else {
        (0..packed_fold_size)
            .into_par_iter()
            .map(compute_iteration)
            .reduce(
                || init_values.clone(),
                |mut acc, sums| {
                    sum_values(&mut acc, sums);
                    acc
                },
            )
    };

    let mut all_evals = vec![];
    for (zs, sum_zs_packed) in all_zs.iter().zip(all_sum_zs_packed) {
        let mut evals = Vec::with_capacity(zs.len());
        for (z_index, z) in zs.iter().enumerate() {
            let mut sum_z = EFPacking::<EF>::to_ext_iter([sum_zs_packed[z_index]]).sum::<EF>();
            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= missing_mul_factor;
            }
            evals.push((PF::<EF>::from_usize(*z), sum_z));
        }
        all_evals.push(evals);
    }

    all_evals
}

fn helper_4<
    const STEP: usize,
    EF: ExtensionField<PF<EF>>, // extension field
    SC: SumcheckComputation<EF>,
>(
    zs: &Vec<usize>,
    folding_factors: &Vec<Vec<PFPacking<EF>>>,
    rows_f: &Vec<Vec<EFPacking<EF>>>,
    rows_ef: &Vec<Vec<EFPacking<EF>>>,
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_mle_eval: Option<EFPacking<EF>>,
    res: &mut Vec<Vec<EFPacking<EF>>>,
) {
    let n = zs.len();
    let evals = (0..n)
        .map(|z_index| {
            let folding_factors_z = &folding_factors[z_index];
            let point_f = rows_f
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<EFPacking<EF>>()
                })
                .collect::<Vec<_>>();
            let point_ef = rows_ef
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_factors_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<EFPacking<EF>>()
                })
                .collect::<Vec<_>>();
            let mut res = computation.eval_packed_extension::<STEP>(
                &point_f,
                &point_ef,
                extra_data,
                alpha_powers,
            );
            if let Some(eq_mle_eval) = eq_mle_eval {
                res *= eq_mle_eval;
            }
            res
        })
        .collect::<Vec<_>>();
    res.push(evals);
}

fn sumcheck_fold_and_compute_packed<
    EF: ExtensionField<PF<EF>>, // extension field
    WPF: PrimeCharacteristicRing + Mul<PFPacking<EF>, Output = WPF> + Copy + Send + Sync + 'static, // witness packed field (either base or extension)
    SC: SumcheckComputation<EF>,
>(
    prev_folding_factors: &[EF],
    multilinears_f: &[&[WPF]],
    multilinears_ef: &[&[EFPacking<EF>]],
    all_zs: &[Vec<usize>],
    skips: usize,
    eq_mle: Option<&[EFPacking<EF>]>,
    all_folding_factors: &[Vec<Vec<PF<EF>>>],
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    missing_mul_factor: Option<EF>,
    compute_fold_size: usize,
    mul: impl Fn(WPF, EF) -> EFPacking<EF> + Sync + Send,
) -> (Vec<Vec<(PF<EF>, EF)>>, MleGroupOwned<EF>, MleGroupOwned<EF>)
where
    EFPacking<EF>: Add<WPF, Output = EFPacking<EF>>,
{
    let prev_folded_size = multilinears_f[0].len() / prev_folding_factors.len();
    let folded_f = (0..multilinears_f.len())
        .map(|_| EFPacking::<EF>::zero_vec(prev_folded_size))
        .collect::<Vec<_>>();
    let folded_ef = (0..multilinears_ef.len())
        .map(|_| EFPacking::<EF>::zero_vec(prev_folded_size))
        .collect::<Vec<_>>();

    let bi_folded = prev_folding_factors.len() == 2;
    if bi_folded {
        assert!(prev_folding_factors[0] == EF::ONE - prev_folding_factors[1],);
    }

    let all_folding_factors = all_folding_factors
        .iter()
        .map(|folding_factors| {
            folding_factors
                .iter()
                .map(|scalars| {
                    scalars
                        .iter()
                        .map(|s| PFPacking::<EF>::from(*s))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let compute_iteration = |i: usize| -> Vec<Vec<EFPacking<EF>>> {
        let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
        let rows_f = multilinears_f
            .iter()
            .enumerate()
            .map(|(j, m)| {
                (0..1 << skips)
                    .map(|k| {
                        let id = i + k * compute_fold_size;
                        let res: EFPacking<EF> = if bi_folded {
                            mul(m[id + prev_folded_size] - m[id], prev_folding_factors[1]) + m[id]
                        } else {
                            prev_folding_factors
                                .iter()
                                .enumerate()
                                .map(|(l, &f)| mul(m[id + l * prev_folded_size], f))
                                .sum()
                        };
                        unsafe {
                            let folded_ptr = folded_f[j].as_ptr() as *mut EFPacking<EF>;
                            *folded_ptr.add(id) = res;
                        }
                        res
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows_ef = multilinears_ef
            .iter()
            .enumerate()
            .map(|(j, m)| {
                (0..1 << skips)
                    .map(|k| {
                        let id = i + k * compute_fold_size;
                        let res: EFPacking<EF> = if bi_folded {
                            Add::<EFPacking<EF>>::add(
                                m[id + prev_folded_size] - m[id] * prev_folding_factors[1],
                                m[id],
                            )
                        } else {
                            prev_folding_factors
                                .iter()
                                .enumerate()
                                .map(|(l, &f)| m[id + l * prev_folded_size] * f)
                                .sum()
                        };
                        unsafe {
                            let folded_ptr = folded_ef[j].as_ptr() as *mut EFPacking<EF>;
                            *folded_ptr.add(id) = res;
                        }
                        res
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut res = Vec::with_capacity(computation.n_steps());
        unroll_match!(SC::N_STEPS, I, {
            helper_4::<I, _, _>(
                &all_zs[I],
                &all_folding_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
                &mut res,
            );
        });

        res
    };

    let init_values = all_zs
        .iter()
        .map(|zs| vec![EFPacking::<EF>::ZERO; zs.len()])
        .collect::<Vec<_>>();
    let sum_values = |current_vals: &mut Vec<Vec<EFPacking<EF>>>,
                      new_vals: Vec<Vec<EFPacking<EF>>>| {
        current_vals
            .iter_mut()
            .zip(new_vals)
            .for_each(|(curr, new)| {
                curr.iter_mut()
                    .zip(new)
                    .for_each(|(curr, new)| *curr += new);
            });
    };

    let all_sum_zs_packed = if compute_fold_size < PARALLEL_THRESHOLD {
        (0..compute_fold_size).fold(init_values, |mut acc, i| {
            sum_values(&mut acc, compute_iteration(i));
            acc
        })
    } else {
        (0..compute_fold_size)
            .into_par_iter()
            .map(compute_iteration)
            .reduce(
                || init_values.clone(),
                |mut acc, sums| {
                    sum_values(&mut acc, sums);
                    acc
                },
            )
    };

    let mut all_evals = vec![];
    for (zs, sum_zs_packed) in all_zs.iter().zip(all_sum_zs_packed) {
        let mut evals = Vec::with_capacity(zs.len());
        for (z_index, z) in zs.iter().enumerate() {
            let mut sum_z = EFPacking::<EF>::to_ext_iter([sum_zs_packed[z_index]]).sum::<EF>();
            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= missing_mul_factor;
            }
            evals.push((PF::<EF>::from_usize(*z), sum_z));
        }
        all_evals.push(evals);
    }

    (
        all_evals,
        MleGroupOwned::ExtensionPacked(folded_f),
        MleGroupOwned::ExtensionPacked(folded_ef),
    )
}
