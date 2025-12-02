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
use std::iter::Sum;
use std::ops::Add;
use std::ops::Mul;
use unroll_macro::unroll_match;

pub trait SumcheckComputation<EF: ExtensionField<PF<EF>>>: Sync + 'static {
    type ExtraData: Send + Sync + 'static;
    const N_STEPS: usize = 1;

    fn degrees(&self) -> Vec<usize>;

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

macro_rules! impl_air_eval {
    ($fn_name:ident, $folder_type:ident, $point_f_ty:ty, $point_ef_ty:ty, $acc_ty:ty, $acc_default:expr) => {
        #[inline(always)]
        fn $fn_name<const STEP: usize>(
            &self,
            point_f: &[$point_f_ty],
            point_ef: &[$point_ef_ty],
            extra_data: &Self::ExtraData,
            alpha_powers: &[EF],
        ) -> $acc_ty {
            let mut folder = $folder_type {
                up_f: &point_f[..self.n_columns_f_air()],
                down_f: &point_f[self.n_columns_f_air()..],
                up_ef: &point_ef[..self.n_columns_ef_air()],
                down_ef: &point_ef[self.n_columns_ef_air()..],
                extra_data,
                alpha_powers: &alpha_powers[self.n_constraints_before_step(STEP)..],
                accumulator: $acc_default,
                constraint_index: 0,
            };
            Air::eval::<_, STEP>(self, &mut folder, extra_data);
            folder.accumulator
        }
    };
}

impl<EF, A> SumcheckComputation<EF> for A
where
    EF: ExtensionField<PF<EF>>,
    A: Send + Sync + Air,
{
    type ExtraData = A::ExtraData;

    impl_air_eval!(eval_base, ConstraintFolder, PF<EF>, EF, EF, EF::ZERO);
    impl_air_eval!(eval_extension, ConstraintFolder, EF, EF, EF, EF::ZERO);
    impl_air_eval!(
        eval_packed_base,
        ConstraintFolderPackedBase,
        PFPacking<EF>,
        EFPacking<EF>,
        EFPacking<EF>,
        Default::default()
    );
    impl_air_eval!(
        eval_packed_extension,
        ConstraintFolderPackedExtension,
        EFPacking<EF>,
        EFPacking<EF>,
        EFPacking<EF>,
        Default::default()
    );

    fn degrees(&self) -> Vec<usize> {
        self.degrees()
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

fn accumulate<T, F, S>(size: usize, init: T, compute: F, sum: S) -> T
where
    T: Clone + Send + Sync,
    F: Fn(usize) -> T + Sync,
    S: Fn(&mut T, T) + Sync + Send,
{
    if size < PARALLEL_THRESHOLD {
        (0..size).fold(init, |mut acc, i| {
            sum(&mut acc, compute(i));
            acc
        })
    } else {
        (0..size).into_par_iter().map(&compute).reduce(
            || init.clone(),
            |mut acc, sums| {
                sum(&mut acc, sums);
                acc
            },
        )
    }
}

fn sum_nested<T: Add<Output = T> + Copy>(current: &mut Vec<Vec<T>>, new: Vec<Vec<T>>) {
    current.iter_mut().zip(new).for_each(|(curr, new)| {
        curr.iter_mut().zip(new).for_each(|(c, n)| *c = *c + n);
    });
}

fn collect_evals<EF, T: Copy, F>(
    all_zs: &[Vec<usize>],
    all_sums: Vec<Vec<T>>,
    missing_mul_factor: Option<EF>,
    extract: F,
) -> Vec<Vec<(PF<EF>, EF)>>
where
    EF: ExtensionField<PF<EF>>,
    F: Fn(T) -> EF,
{
    all_zs
        .iter()
        .zip(all_sums)
        .map(|(zs, sum_zs)| {
            zs.iter()
                .enumerate()
                .map(|(z_index, z)| {
                    let mut sum_z = extract(sum_zs[z_index]);
                    if let Some(factor) = missing_mul_factor {
                        sum_z *= factor;
                    }
                    (PF::<EF>::from_usize(*z), sum_z)
                })
                .collect()
        })
        .collect()
}

fn eval_step_scalar<const STEP: usize, EF, IF, SC>(
    zs: &[usize],
    folding_factors: &[Vec<PF<EF>>],
    rows_f: &[Vec<IF>],
    rows_ef: &[Vec<EF>],
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_mle_eval: Option<EF>,
) -> Vec<EF>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>> + Mul<PF<EF>, Output = IF> + Sum + Copy,
    SC: SumcheckComputation<EF>,
{
    (0..zs.len())
        .map(|z_index| {
            let ff = &folding_factors[z_index];
            let point_f: Vec<IF> = rows_f
                .iter()
                .map(|row| row.iter().zip(ff).map(|(&x, &s)| x * s).sum())
                .collect();
            let point_ef: Vec<EF> = rows_ef
                .iter()
                .map(|row| row.iter().zip(ff).map(|(&x, &s)| x * s).sum())
                .collect();

            let mut res = if TypeId::of::<IF>() == TypeId::of::<PF<EF>>() {
                let pf = unsafe { std::mem::transmute::<Vec<IF>, Vec<PF<EF>>>(point_f) };
                computation.eval_base::<STEP>(&pf, &point_ef, extra_data, alpha_powers)
            } else {
                let ef = unsafe { std::mem::transmute::<Vec<IF>, Vec<EF>>(point_f) };
                computation.eval_extension::<STEP>(&ef, &point_ef, extra_data, alpha_powers)
            };
            if let Some(eq) = eq_mle_eval {
                res *= eq;
            }
            res
        })
        .collect()
}

fn eval_step_packed<const STEP: usize, EF, WPF, SC>(
    zs: &[usize],
    folding_factors: &[Vec<PFPacking<EF>>],
    rows_f: &[Vec<WPF>],
    rows_ef: &[Vec<EFPacking<EF>>],
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_mle_eval: Option<EFPacking<EF>>,
) -> Vec<EFPacking<EF>>
where
    EF: ExtensionField<PF<EF>>,
    WPF: PrimeCharacteristicRing + Mul<PFPacking<EF>, Output = WPF> + Copy + Sum + 'static,
    SC: SumcheckComputation<EF>,
{
    (0..zs.len())
        .map(|z_index| {
            let ff = &folding_factors[z_index];
            let point_f: Vec<WPF> = rows_f
                .iter()
                .map(|row| row.iter().zip(ff).map(|(&x, &s)| x * s).sum())
                .collect();
            let point_ef: Vec<EFPacking<EF>> = rows_ef
                .iter()
                .map(|row| row.iter().zip(ff).map(|(&x, &s)| x * s).sum())
                .collect();

            let mut res = if TypeId::of::<WPF>() == TypeId::of::<PFPacking<EF>>() {
                let pf = unsafe { std::mem::transmute::<Vec<WPF>, Vec<PFPacking<EF>>>(point_f) };
                computation.eval_packed_base::<STEP>(&pf, &point_ef, extra_data, alpha_powers)
            } else {
                let ef = unsafe { std::mem::transmute::<Vec<WPF>, Vec<EFPacking<EF>>>(point_f) };
                computation.eval_packed_extension::<STEP>(&ef, &point_ef, extra_data, alpha_powers)
            };
            if let Some(eq) = eq_mle_eval {
                res *= eq;
            }
            res
        })
        .collect()
}

fn extract_rows<T: Copy>(
    multilinears: &[&[T]],
    i: usize,
    fold_size: usize,
    skips: usize,
) -> Vec<Vec<T>> {
    multilinears
        .iter()
        .map(|m| (0..1 << skips).map(|j| m[i + j * fold_size]).collect())
        .collect()
}

fn try_special_case_compute<'a, EF, SC>(
    group_f: &MleGroupRef<'a, EF>,
    group_ef: &MleGroupRef<'a, EF>,
    params: &SumcheckComputeParams<'a, EF, SC>,
) -> Option<Vec<Vec<(PF<EF>, EF)>>>
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF> + 'static,
{
    let SumcheckComputeParams {
        eq_mle,
        first_eq_factor,
        alpha_powers,
        missing_mul_factor,
        sums,
        ..
    } = params;

    if TypeId::of::<SC>() == TypeId::of::<ProductComputation>() && eq_mle.is_none() {
        assert!(missing_mul_factor.is_none() && alpha_powers.is_empty());
        assert_eq!(group_f.n_columns(), 2);
        assert_eq!(group_ef.n_columns(), 0);
        assert_eq!(sums.len(), 1);

        let poly = match group_f {
            MleGroupRef::Extension(m) => {
                compute_product_sumcheck_polynomial(&m[0], &m[1], sums[0], |e| vec![e])
            }
            MleGroupRef::ExtensionPacked(m) => {
                compute_product_sumcheck_polynomial(&m[0], &m[1], sums[0], |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                })
            }
            _ => unimplemented!(),
        };
        return Some(vec![vec![
            (PF::<EF>::ZERO, poly.coeffs[0]),
            (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
        ]]);
    }

    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<2>>() {
        assert!(eq_mle.is_some());
        assert_eq!(group_f.n_columns(), 4);
        assert_eq!(group_ef.n_columns(), 0);
        assert_eq!(sums.len(), 1);

        let poly = match group_f {
            MleGroupRef::Extension(m) => compute_gkr_quotient_sumcheck_polynomial(
                &m[0],
                &m[1],
                &m[2],
                &m[3],
                alpha_powers[1],
                first_eq_factor.unwrap(),
                eq_mle.unwrap().as_extension().unwrap(),
                missing_mul_factor.unwrap_or(EF::ONE),
                sums[0],
                |e| vec![e],
            ),
            MleGroupRef::ExtensionPacked(m) => compute_gkr_quotient_sumcheck_polynomial(
                &m[0],
                &m[1],
                &m[2],
                &m[3],
                alpha_powers[1],
                first_eq_factor.unwrap(),
                eq_mle.unwrap().as_extension_packed().unwrap(),
                missing_mul_factor.unwrap_or(EF::ONE),
                sums[0],
                |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
            ),
            _ => unimplemented!(),
        };
        return Some(vec![vec![
            (PF::<EF>::ZERO, poly.coeffs[0]),
            (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
        ]]);
    }

    None
}

fn try_special_case_fold_and_compute<'a, EF, SC>(
    prev_folding_factors: &[EF],
    group_f: &MleGroupRef<'a, EF>,
    group_ef: &MleGroupRef<'a, EF>,
    params: &SumcheckComputeParams<'a, EF, SC>,
) -> Option<(Vec<Vec<(PF<EF>, EF)>>, MleGroupOwned<EF>, MleGroupOwned<EF>)>
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF> + 'static,
{
    let SumcheckComputeParams {
        eq_mle,
        first_eq_factor,
        alpha_powers,
        missing_mul_factor,
        sums,
        ..
    } = params;

    let is_bi_fold = prev_folding_factors.len() == 2
        && prev_folding_factors[0] == EF::ONE - prev_folding_factors[1];

    if !is_bi_fold {
        return None;
    }

    let prev_folding_factor = prev_folding_factors[1];

    if TypeId::of::<SC>() == TypeId::of::<ProductComputation>() && eq_mle.is_none() {
        assert!(missing_mul_factor.is_none() && alpha_powers.is_empty());
        assert_eq!(group_f.n_columns(), 2);
        assert_eq!(group_ef.n_columns(), 0);
        assert_eq!(sums.len(), 1);

        let (poly, folded_f) = match group_f {
            MleGroupRef::Extension(m) => {
                let (p, f) = fold_and_compute_product_sumcheck_polynomial(
                    &m[0],
                    &m[1],
                    prev_folding_factor,
                    sums[0],
                    |e| vec![e],
                );
                (p, MleGroupOwned::Extension(f))
            }
            MleGroupRef::ExtensionPacked(m) => {
                let (p, f) = fold_and_compute_product_sumcheck_polynomial(
                    &m[0],
                    &m[1],
                    prev_folding_factor,
                    sums[0],
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                (p, MleGroupOwned::ExtensionPacked(f))
            }
            _ => unimplemented!(),
        };
        let folded_ef = MleGroupOwned::empty(true, folded_f.is_packed());
        return Some((
            vec![vec![
                (PF::<EF>::ZERO, poly.coeffs[0]),
                (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
            ]],
            folded_f,
            folded_ef,
        ));
    }

    if TypeId::of::<SC>() == TypeId::of::<GKRQuotientComputation<2>>() {
        assert!(eq_mle.is_some());
        assert_eq!(group_f.n_columns(), 4);
        assert_eq!(group_ef.n_columns(), 0);
        assert_eq!(sums.len(), 1);

        let (poly, folded_f) = match group_f {
            MleGroupRef::Extension(m) => {
                let (p, f) = fold_and_compute_gkr_quotient_sumcheck_polynomial(
                    prev_folding_factor,
                    &m[0],
                    &m[1],
                    &m[2],
                    &m[3],
                    alpha_powers[1],
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sums[0],
                    |e| vec![e],
                );
                (p, MleGroupOwned::Extension(f))
            }
            MleGroupRef::ExtensionPacked(m) => {
                let (p, f) = fold_and_compute_gkr_quotient_sumcheck_polynomial(
                    prev_folding_factor,
                    &m[0],
                    &m[1],
                    &m[2],
                    &m[3],
                    alpha_powers[1],
                    first_eq_factor.unwrap(),
                    eq_mle.unwrap().as_extension_packed().unwrap(),
                    missing_mul_factor.unwrap_or(EF::ONE),
                    sums[0],
                    |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
                );
                (p, MleGroupOwned::ExtensionPacked(f))
            }
            _ => unimplemented!(),
        };
        let folded_ef = MleGroupOwned::empty(true, folded_f.is_packed());
        return Some((
            vec![vec![
                (PF::<EF>::ZERO, poly.coeffs[0]),
                (PF::<EF>::TWO, poly.evaluate(EF::TWO)),
            ]],
            folded_f,
            folded_ef,
        ));
    }

    None
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
    if let Some(result) = try_special_case_compute(group_f, group_ef, &params) {
        return result;
    }

    let SumcheckComputeParams {
        skips,
        eq_mle,
        folding_factors,
        computation,
        extra_data,
        alpha_powers,
        missing_mul_factor,
        ..
    } = params;
    let fold_size = 1 << (group_f.n_vars() - skips);
    let packed_fold_size = if group_f.is_packed() {
        fold_size / packing_width::<EF>()
    } else {
        fold_size
    };

    match group_f {
        MleGroupRef::ExtensionPacked(mf) => sumcheck_compute_packed::<EF, EFPacking<EF>, _>(
            mf,
            group_ef.as_extension_packed().unwrap(),
            all_zs,
            skips,
            eq_mle.map(|e| e.as_extension_packed().unwrap()),
            folding_factors,
            computation,
            extra_data,
            alpha_powers,
            missing_mul_factor,
            packed_fold_size,
        ),
        MleGroupRef::BasePacked(mf) => sumcheck_compute_packed::<EF, PFPacking<EF>, _>(
            mf,
            group_ef.as_extension_packed().unwrap(),
            all_zs,
            skips,
            eq_mle.map(|e| e.as_extension_packed().unwrap()),
            folding_factors,
            computation,
            extra_data,
            alpha_powers,
            missing_mul_factor,
            packed_fold_size,
        ),
        MleGroupRef::Base(mf) => sumcheck_compute_scalar::<EF, PF<EF>, _>(
            mf,
            group_ef.as_extension().unwrap(),
            all_zs,
            skips,
            eq_mle.map(|e| e.as_extension().unwrap()),
            folding_factors,
            computation,
            extra_data,
            alpha_powers,
            missing_mul_factor,
            fold_size,
        ),
        MleGroupRef::Extension(mf) => sumcheck_compute_scalar::<EF, EF, _>(
            mf,
            group_ef.as_extension().unwrap(),
            all_zs,
            skips,
            eq_mle.map(|e| e.as_extension().unwrap()),
            folding_factors,
            computation,
            extra_data,
            alpha_powers,
            missing_mul_factor,
            fold_size,
        ),
    }
}

fn sumcheck_compute_scalar<EF, IF, SC>(
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
) -> Vec<Vec<(PF<EF>, EF)>>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>> + Mul<PF<EF>, Output = IF> + Sum + Copy,
    SC: SumcheckComputation<EF>,
{
    let compute_iteration = |i: usize| -> Vec<Vec<EF>> {
        let eq_mle_eval = eq_mle.map(|e| e[i]);
        let rows_f = extract_rows(multilinears_f, i, fold_size, skips);
        let rows_ef = extract_rows(multilinears_ef, i, fold_size, skips);

        let mut res = Vec::with_capacity(computation.n_steps());
        unroll_match!(SC::N_STEPS, I, {
            res.push(eval_step_scalar::<I, EF, IF, SC>(
                &all_zs[I],
                &all_folding_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
            ));
        });
        res
    };

    let init = all_zs.iter().map(|zs| vec![EF::ZERO; zs.len()]).collect();
    let all_sum_zs = accumulate(fold_size, init, compute_iteration, sum_nested);
    collect_evals(all_zs, all_sum_zs, missing_mul_factor, |e| e)
}

fn sumcheck_compute_packed<EF, WPF, SC>(
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
) -> Vec<Vec<(PF<EF>, EF)>>
where
    EF: ExtensionField<PF<EF>>,
    WPF: PrimeCharacteristicRing
        + Mul<PFPacking<EF>, Output = WPF>
        + Copy
        + Send
        + Sync
        + Sum
        + 'static,
    SC: SumcheckComputation<EF>,
{
    let packed_factors: Vec<Vec<Vec<PFPacking<EF>>>> = all_folding_factors
        .iter()
        .map(|ff| {
            ff.iter()
                .map(|s| s.iter().map(|&x| PFPacking::<EF>::from(x)).collect())
                .collect()
        })
        .collect();

    let compute_iteration = |i: usize| -> Vec<Vec<EFPacking<EF>>> {
        let eq_mle_eval = eq_mle.map(|e| e[i]);
        let rows_f = extract_rows(multilinears_f, i, packed_fold_size, skips);
        let rows_ef = extract_rows(multilinears_ef, i, packed_fold_size, skips);

        let mut res = Vec::with_capacity(computation.n_steps());
        unroll_match!(SC::N_STEPS, I, {
            res.push(eval_step_packed::<I, EF, WPF, SC>(
                &all_zs[I],
                &packed_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
            ));
        });
        res
    };

    let init = all_zs
        .iter()
        .map(|zs| vec![EFPacking::<EF>::ZERO; zs.len()])
        .collect();
    let all_sum_zs = accumulate(packed_fold_size, init, compute_iteration, sum_nested);
    collect_evals(all_zs, all_sum_zs, missing_mul_factor, |e| {
        EFPacking::<EF>::to_ext_iter([e]).sum()
    })
}

pub fn fold_and_sumcheck_compute<'a, EF: ExtensionField<PF<EF>>, SC: SumcheckComputation<EF>>(
    prev_folding_factors: &[EF],
    group_f: &MleGroupRef<'a, EF>,
    group_ef: &MleGroupRef<'a, EF>,
    params: SumcheckComputeParams<'a, EF, SC>,
    all_zs: &[Vec<usize>],
) -> (Vec<Vec<(PF<EF>, EF)>>, MleGroupOwned<EF>, MleGroupOwned<EF>) {
    if let Some(result) =
        try_special_case_fold_and_compute(prev_folding_factors, group_f, group_ef, &params)
    {
        return result;
    }

    let SumcheckComputeParams {
        skips,
        eq_mle,
        folding_factors,
        computation,
        extra_data,
        alpha_powers,
        missing_mul_factor,
        ..
    } = params;
    let fold_size = 1 << (group_f.n_vars() - skips - log2_strict_usize(prev_folding_factors.len()));
    let compute_fold_size = if group_f.is_packed() {
        fold_size / packing_width::<EF>()
    } else {
        fold_size
    };

    match group_f {
        MleGroupRef::ExtensionPacked(mf) => {
            sumcheck_fold_and_compute_packed::<EF, EFPacking<EF>, _>(
                prev_folding_factors,
                mf,
                group_ef.as_extension_packed().unwrap(),
                all_zs,
                skips,
                eq_mle.map(|e| e.as_extension_packed().unwrap()),
                folding_factors,
                computation,
                extra_data,
                alpha_powers,
                missing_mul_factor,
                compute_fold_size,
                |wpf, ef| wpf * ef,
            )
        }
        MleGroupRef::BasePacked(mf) => sumcheck_fold_and_compute_packed::<EF, PFPacking<EF>, _>(
            prev_folding_factors,
            mf,
            group_ef.as_extension_packed().unwrap(),
            all_zs,
            skips,
            eq_mle.map(|e| e.as_extension_packed().unwrap()),
            folding_factors,
            computation,
            extra_data,
            alpha_powers,
            missing_mul_factor,
            compute_fold_size,
            |wpf, ef| EFPacking::<EF>::from(ef) * wpf,
        ),
        MleGroupRef::Base(mf) => sumcheck_fold_and_compute_scalar::<EF, PF<EF>, _>(
            prev_folding_factors,
            mf,
            group_ef.as_extension().unwrap(),
            all_zs,
            skips,
            eq_mle.map(|e| e.as_extension().unwrap()),
            folding_factors,
            computation,
            extra_data,
            alpha_powers,
            missing_mul_factor,
            fold_size,
        ),
        MleGroupRef::Extension(mf) => sumcheck_fold_and_compute_scalar::<EF, EF, _>(
            prev_folding_factors,
            mf,
            group_ef.as_extension().unwrap(),
            all_zs,
            skips,
            eq_mle.map(|e| e.as_extension().unwrap()),
            folding_factors,
            computation,
            extra_data,
            alpha_powers,
            missing_mul_factor,
            fold_size,
        ),
    }
}

fn sumcheck_fold_and_compute_scalar<EF, IF, SC>(
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
) -> (Vec<Vec<(PF<EF>, EF)>>, MleGroupOwned<EF>, MleGroupOwned<EF>)
where
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>> + Mul<PF<EF>, Output = IF> + Sum + Copy,
    SC: SumcheckComputation<EF>,
{
    let bi_folded = prev_folding_factors.len() == 2
        && prev_folding_factors[0] == EF::ONE - prev_folding_factors[1];
    let prev_folded_size = multilinears_f[0].len() / prev_folding_factors.len();

    let folded_f: Vec<Vec<EF>> = (0..multilinears_f.len())
        .map(|_| EF::zero_vec(prev_folded_size))
        .collect();
    let folded_ef: Vec<Vec<EF>> = (0..multilinears_ef.len())
        .map(|_| EF::zero_vec(prev_folded_size))
        .collect();

    let compute_iteration = |i: usize| -> Vec<Vec<EF>> {
        let eq_mle_eval = eq_mle.map(|e| e[i]);

        let rows_f: Vec<Vec<EF>> = multilinears_f
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
                            *(folded_f[j].as_ptr() as *mut EF).add(id) = res;
                        }
                        res
                    })
                    .collect()
            })
            .collect();

        let rows_ef: Vec<Vec<EF>> = multilinears_ef
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
                            *(folded_ef[j].as_ptr() as *mut EF).add(id) = res;
                        }
                        res
                    })
                    .collect()
            })
            .collect();

        let mut res = Vec::with_capacity(computation.n_steps());
        unroll_match!(SC::N_STEPS, I, {
            res.push(eval_step_scalar::<I, EF, EF, SC>(
                &all_zs[I],
                &all_folding_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
            ));
        });
        res
    };

    let init = all_zs.iter().map(|zs| vec![EF::ZERO; zs.len()]).collect();
    let all_sum_zs = accumulate(compute_fold_size, init, compute_iteration, sum_nested);
    let evals = collect_evals(all_zs, all_sum_zs, missing_mul_factor, |e| e);

    (
        evals,
        MleGroupOwned::Extension(folded_f),
        MleGroupOwned::Extension(folded_ef),
    )
}

fn sumcheck_fold_and_compute_packed<EF, WPF, SC>(
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
    EF: ExtensionField<PF<EF>>,
    WPF: PrimeCharacteristicRing
        + Mul<PFPacking<EF>, Output = WPF>
        + Copy
        + Send
        + Sync
        + Sum
        + 'static,
    EFPacking<EF>: Add<WPF, Output = EFPacking<EF>>,
    SC: SumcheckComputation<EF>,
{
    let bi_folded = prev_folding_factors.len() == 2
        && prev_folding_factors[0] == EF::ONE - prev_folding_factors[1];
    let prev_folded_size = multilinears_f[0].len() / prev_folding_factors.len();

    let folded_f: Vec<Vec<EFPacking<EF>>> = (0..multilinears_f.len())
        .map(|_| EFPacking::<EF>::zero_vec(prev_folded_size))
        .collect();
    let folded_ef: Vec<Vec<EFPacking<EF>>> = (0..multilinears_ef.len())
        .map(|_| EFPacking::<EF>::zero_vec(prev_folded_size))
        .collect();

    let packed_factors: Vec<Vec<Vec<PFPacking<EF>>>> = all_folding_factors
        .iter()
        .map(|ff| {
            ff.iter()
                .map(|s| s.iter().map(|&x| PFPacking::<EF>::from(x)).collect())
                .collect()
        })
        .collect();

    let compute_iteration = |i: usize| -> Vec<Vec<EFPacking<EF>>> {
        let eq_mle_eval = eq_mle.map(|e| e[i]);

        let rows_f: Vec<Vec<EFPacking<EF>>> = multilinears_f
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
                            *(folded_f[j].as_ptr() as *mut EFPacking<EF>).add(id) = res;
                        }
                        res
                    })
                    .collect()
            })
            .collect();

        let rows_ef: Vec<Vec<EFPacking<EF>>> = multilinears_ef
            .iter()
            .enumerate()
            .map(|(j, m)| {
                (0..1 << skips)
                    .map(|k| {
                        let id = i + k * compute_fold_size;
                        let res: EFPacking<EF> = if bi_folded {
                            <EFPacking<EF> as Add<EFPacking<EF>>>::add(
                                (m[id + prev_folded_size] - m[id]) * prev_folding_factors[1],
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
                            *(folded_ef[j].as_ptr() as *mut EFPacking<EF>).add(id) = res;
                        }
                        res
                    })
                    .collect()
            })
            .collect();

        let mut res = Vec::with_capacity(computation.n_steps());
        unroll_match!(SC::N_STEPS, I, {
            res.push(eval_step_packed::<I, EF, EFPacking<EF>, SC>(
                &all_zs[I],
                &packed_factors[I],
                &rows_f,
                &rows_ef,
                computation,
                extra_data,
                alpha_powers,
                eq_mle_eval,
            ));
        });
        res
    };

    let init = all_zs
        .iter()
        .map(|zs| vec![EFPacking::<EF>::ZERO; zs.len()])
        .collect();
    let all_sum_zs = accumulate(compute_fold_size, init, compute_iteration, sum_nested);
    let evals = collect_evals(all_zs, all_sum_zs, missing_mul_factor, |e| {
        EFPacking::<EF>::to_ext_iter([e]).sum()
    });

    (
        evals,
        MleGroupOwned::ExtensionPacked(folded_f),
        MleGroupOwned::ExtensionPacked(folded_ef),
    )
}
