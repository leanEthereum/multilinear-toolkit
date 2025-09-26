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

use crate::ProductComputation;
use crate::compute_product_sumcheck_polynomial;

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
) -> Vec<(PF<EF>, EF)>
where
    SC: SumcheckComputation<PF<EF>, EF> + SumcheckComputation<EF, EF> + 'static,
    SCP: SumcheckComputationPacked<EF>,
{
    let SumcheckComputeParams {
        zs,
        skips,
        eq_mle,
        folding_scalars,
        computation,
        computation_packed,
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

        match group {
            MleGroupRef::Extension(multilinears) => {
                let pol_0 = &multilinears[0];
                let pol_1 = &multilinears[1];
                let [c0, c1, c2] = compute_product_sumcheck_polynomial(pol_0, pol_1, sum);
                let eval_0 = c0;
                let eval_1 = c0 + c1 + c2;
                let eval_2 = eval_1 + c1 + c2 + c2.double();
                assert_eq!(zs, &[0, 2]);
                return vec![(PF::<EF>::ZERO, eval_0), (PF::<EF>::TWO, eval_2)];
            }
            _ => unimplemented!(),
        }
    }

    match group {
        MleGroupRef::ExtensionPacked(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());
            let all_sums =
                unsafe { uninitialized_vec::<EFPacking<EF>>(zs.len() * packed_fold_size) }; // sums for zs[0], sums for zs[1], ...
            (0..packed_fold_size).into_par_iter().for_each(|i| {
                let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
                let rows = multilinears
                    .iter()
                    .map(|m| {
                        (0..1 << skips)
                            .map(|j| m[i + j * packed_fold_size])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                for (z_index, folding_scalars_z) in folding_scalars.iter().enumerate() {
                    let point = rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(folding_scalars_z.iter())
                                .map(|(x, s)| *x * PFPacking::<EF>::from(*s))
                                .sum::<EFPacking<EF>>()
                        })
                        .collect::<Vec<_>>();

                    let mut res =
                        computation_packed.eval_packed_extension(&point, batching_scalars);
                    if let Some(eq_mle_eval) = eq_mle_eval {
                        res *= eq_mle_eval;
                    }

                    unsafe {
                        let sum_ptr = all_sums.as_ptr() as *mut EFPacking<EF>;
                        *sum_ptr.add(z_index * packed_fold_size + i) = res;
                    }
                }
            });

            let mut evals = vec![];
            for (z_index, z) in zs.iter().enumerate() {
                let mut sum_z = all_sums
                    [z_index * packed_fold_size..(z_index + 1) * packed_fold_size]
                    .par_iter()
                    .copied()
                    .sum::<EFPacking<EF>>();
                if let Some(missing_mul_factor) = missing_mul_factor {
                    sum_z *= missing_mul_factor;
                }
                evals.push((
                    PF::<EF>::from_usize(*z),
                    EFPacking::<EF>::to_ext_iter([sum_z]).sum::<EF>(),
                ));
            }
            evals
        }
        MleGroupRef::BasePacked(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension_packed().unwrap());

            let all_sums =
                unsafe { uninitialized_vec::<EFPacking<EF>>(zs.len() * packed_fold_size) }; // sums for zs[0], sums for zs[1], ...
            (0..packed_fold_size).into_par_iter().for_each(|i| {
                let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
                let rows = multilinears
                    .iter()
                    .map(|m| {
                        (0..1 << skips)
                            .map(|j| m[i + j * packed_fold_size])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                for (z_index, folding_scalars_z) in folding_scalars.iter().enumerate() {
                    let point = rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(folding_scalars_z.iter())
                                .map(|(x, s)| *x * *s)
                                .sum::<PFPacking<EF>>()
                        })
                        .collect::<Vec<_>>();

                    let mut res = computation_packed.eval_packed_base(&point, batching_scalars);
                    if let Some(eq_mle_eval) = eq_mle_eval {
                        res *= eq_mle_eval;
                    }

                    unsafe {
                        let sum_ptr = all_sums.as_ptr() as *mut EFPacking<EF>;
                        *sum_ptr.add(z_index * packed_fold_size + i) = res;
                    }
                }
            });

            let mut evals = vec![];
            for (z_index, z) in zs.iter().enumerate() {
                let sum_z_packed = all_sums
                    [z_index * packed_fold_size..(z_index + 1) * packed_fold_size]
                    .par_iter()
                    .copied()
                    .sum::<EFPacking<EF>>();
                let mut sum_z = EFPacking::<EF>::to_ext_iter([sum_z_packed]).sum::<EF>();
                if let Some(missing_mul_factor) = missing_mul_factor {
                    sum_z *= missing_mul_factor;
                }
                evals.push((PF::<EF>::from_usize(*z), sum_z));
            }
            evals
        }
        MleGroupRef::Base(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap().as_slice());
            sumcheck_compute_not_packed(
                multilinears,
                SumcheckComputeNotPackedParams {
                    zs,
                    skips,
                    eq_mle,
                    folding_scalars,
                    computation,
                    batching_scalars,
                    missing_mul_factor,
                    fold_size,
                },
            )
        }
        MleGroupRef::Extension(multilinears) => {
            let eq_mle = eq_mle.map(|eq_mle| eq_mle.as_extension().unwrap().as_slice());
            sumcheck_compute_not_packed(
                multilinears,
                SumcheckComputeNotPackedParams {
                    zs,
                    skips,
                    eq_mle,
                    folding_scalars,
                    computation,
                    batching_scalars,
                    missing_mul_factor,
                    fold_size,
                },
            )
        }
    }
}

#[derive(Debug)]
pub struct SumcheckComputeParams<'a, EF: ExtensionField<PF<EF>>, SC, SCP> {
    pub zs: &'a [usize],
    pub skips: usize,
    pub eq_mle: Option<&'a Mle<EF>>,
    pub folding_scalars: &'a [Vec<PF<EF>>],
    pub computation: &'a SC,
    pub computation_packed: &'a SCP,
    pub batching_scalars: &'a [EF],
    pub missing_mul_factor: Option<EF>,
    pub sum: EF,
}

#[derive(Debug)]
pub struct SumcheckComputeNotPackedParams<'a, EF, SC>
where
    EF: ExtensionField<PF<EF>>,
{
    pub zs: &'a [usize],
    pub skips: usize,
    pub eq_mle: Option<&'a [EF]>,
    pub folding_scalars: &'a [Vec<PF<EF>>],
    pub computation: &'a SC,
    pub batching_scalars: &'a [EF],
    pub missing_mul_factor: Option<EF>,
    pub fold_size: usize,
}

fn sumcheck_compute_not_packed<
    EF: ExtensionField<PF<EF>> + ExtensionField<IF>,
    IF: ExtensionField<PF<EF>>,
    SC,
>(
    multilinears: &[&[IF]],
    params: SumcheckComputeNotPackedParams<'_, EF, SC>,
) -> Vec<(PF<EF>, EF)>
where
    SC: SumcheckComputation<IF, EF>,
{
    let SumcheckComputeNotPackedParams {
        zs,
        skips,
        eq_mle,
        folding_scalars,
        computation,
        batching_scalars,
        missing_mul_factor,
        fold_size,
    } = params;

    let all_sums = unsafe { uninitialized_vec::<EF>(zs.len() * fold_size) }; // sums for zs[0], sums for zs[1], ...
    (0..fold_size).into_par_iter().for_each(|i| {
        let eq_mle_eval = eq_mle.as_ref().map(|eq_mle| eq_mle[i]);
        let rows = multilinears
            .iter()
            .map(|m| {
                (0..1 << skips)
                    .map(|j| m[i + j * fold_size])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (z_index, folding_scalars_z) in folding_scalars.iter().enumerate() {
            let point = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_scalars_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<IF>()
                })
                .collect::<Vec<_>>();
            unsafe {
                let sum_ptr = all_sums.as_ptr() as *mut EF;
                let mut res = computation.eval(&point, batching_scalars);
                if let Some(eq_mle_eval) = eq_mle_eval {
                    res *= eq_mle_eval;
                }
                *sum_ptr.add(z_index * fold_size + i) = res;
            }
        }
    });
    let mut evals = vec![];
    for (z_index, z) in zs.iter().enumerate() {
        let mut sum_z = all_sums[z_index * fold_size..(z_index + 1) * fold_size]
            .par_iter()
            .copied()
            .sum::<EF>();
        if let Some(missing_mul_factor) = missing_mul_factor {
            sum_z *= missing_mul_factor;
        }
        evals.push((PF::<EF>::from_usize(*z), sum_z));
    }
    evals
}
