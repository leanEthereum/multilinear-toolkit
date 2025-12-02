use backend::*;
use fiat_shamir::*;
use itertools::Itertools;
use p3_field::ExtensionField;
use p3_field::PrimeCharacteristicRing;
use p3_util::log2_strict_usize;

use crate::*;

#[allow(clippy::too_many_arguments)]
pub fn sumcheck_prove<'a, EF, SC, M: Into<MleGroup<'a, EF>>>(
    skip: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears_f: M,
    multilinears_ef: Option<M>,
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_factor: Option<(Vec<EF>, Option<MleOwned<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    sum: EF,
    store_intermediate_foldings: bool,
) -> (MultilinearPoint<EF>, Vec<EF>, EF)
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF>,
{
    sumcheck_prove_custom(
        skip,
        multilinears_f,
        multilinears_ef,
        None,
        computation,
        extra_data,
        alpha_powers,
        eq_factor,
        vec![is_zerofier],
        prover_state,
        vec![sum],
        store_intermediate_foldings,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sumcheck_prove_custom<'a, EF, SC, M: Into<MleGroup<'a, EF>>>(
    skip: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears_f: M,
    multilinears_ef: Option<M>,
    prev_folding_factors: Option<Vec<EF>>,
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    eq_factor: Option<(Vec<EF>, Option<MleOwned<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    is_zerofier: Vec<bool>,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    sums: Vec<EF>,
    store_intermediate_foldings: bool,
) -> (MultilinearPoint<EF>, Vec<EF>, EF)
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF>,
{
    let multilinears_f: MleGroup<'a, EF> = multilinears_f.into();
    let multilinears_ef: MleGroup<'a, EF> = match multilinears_ef {
        Some(m) => m.into(),
        None => MleGroupOwned::empty(true, multilinears_f.is_packed()).into(),
    };
    let mut n_rounds = multilinears_f.by_ref().n_vars() + 1 - skip;
    if let Some(prev_folding_factors) = &prev_folding_factors {
        n_rounds -= log2_strict_usize(prev_folding_factors.len());
    }
    let (challenges, final_folds_f, final_folds_ef, final_sum) = sumcheck_prove_many_rounds(
        skip,
        multilinears_f,
        Some(multilinears_ef),
        prev_folding_factors,
        computation,
        extra_data,
        alpha_powers,
        eq_factor,
        is_zerofier,
        prover_state,
        sums,
        None,
        n_rounds,
        store_intermediate_foldings,
    );

    let final_folds = [final_folds_f, final_folds_ef]
        .into_iter()
        .map(|mle| {
            mle.by_ref()
                .as_extension()
                .unwrap()
                .iter()
                .map(|m| {
                    assert_eq!(m.len(), 1);
                    m[0]
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    (challenges, final_folds, final_sum)
}

#[allow(clippy::too_many_arguments)]
pub fn sumcheck_prove_many_rounds<'a, EF, SC, M: Into<MleGroup<'a, EF>>>(
    mut skip: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears_f: M,
    multilinears_ef: Option<M>,
    mut prev_folding_factors: Option<Vec<EF>>,
    computation: &SC,
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    mut eq_factor: Option<(Vec<EF>, Option<MleOwned<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    mut all_zerofiers: Vec<bool>,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut sums: Vec<EF>,
    mut missing_mul_factors: Option<EF>,
    n_rounds: usize,
    store_intermediate_foldings: bool,
) -> (
    MultilinearPoint<EF>,
    MleGroupOwned<EF>,
    MleGroupOwned<EF>,
    EF,
)
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF>,
{
    let mut multilinears_f: MleGroup<'a, EF> = multilinears_f.into();
    let mut multilinears_ef: MleGroup<'a, EF> = match multilinears_ef {
        Some(m) => m.into(),
        None => MleGroupOwned::empty(true, multilinears_f.is_packed()).into(),
    };
    assert_eq!(multilinears_f.is_packed(), multilinears_ef.is_packed());

    let mut eq_factor: Option<(Vec<EF>, MleOwned<EF>)> =
        eq_factor.take().map(|(eq_point, eq_mle)| {
            let eq_mle = eq_mle.unwrap_or_else(|| {
                let eval_eq_ext = eval_eq(&eq_point[1..]);
                if multilinears_f.by_ref().is_packed() {
                    MleOwned::ExtensionPacked(pack_extension(&eval_eq_ext))
                } else {
                    MleOwned::Extension(eval_eq_ext)
                }
            });
            (eq_point, eq_mle)
        });

    let mut n_vars = multilinears_f.by_ref().n_vars();
    if let Some(prev_folding_factors) = &prev_folding_factors {
        n_vars -= log2_strict_usize(prev_folding_factors.len());
    }
    if let Some((eq_point, eq_mle)) = &eq_factor {
        assert_eq!(eq_point.len(), n_vars + 1 - skip);
        assert_eq!(eq_mle.by_ref().n_vars(), eq_point.len() - 1);
        if eq_mle.by_ref().is_packed() && !multilinears_f.is_packed() {
            assert!(eq_point.len() < packing_log_width::<EF>());
            multilinears_f = multilinears_f.by_ref().unpack().as_owned_or_clone().into();
            multilinears_ef = multilinears_ef.by_ref().unpack().as_owned_or_clone().into();
        }
    }

    let mut challenges = Vec::new();
    for _ in 0..n_rounds {
        // If Packing is enabled, and there are too little variables, we unpack everything:
        if multilinears_f.by_ref().is_packed() && n_vars <= 1 + packing_log_width::<EF>() {
            // unpack
            multilinears_f = multilinears_f.by_ref().unpack().as_owned_or_clone().into();
            multilinears_ef = multilinears_ef.by_ref().unpack().as_owned_or_clone().into();

            if let Some((_, eq_mle)) = &mut eq_factor {
                *eq_mle = eq_mle.by_ref().unpack().as_owned_or_clone();
            }
        }

        let ps = compute_and_send_polynomial(
            skip,
            &mut multilinears_f,
            &mut multilinears_ef,
            prev_folding_factors,
            computation,
            &eq_factor,
            extra_data,
            alpha_powers,
            &all_zerofiers,
            prover_state,
            &sums,
            missing_mul_factors,
        );
        let challenge = prover_state.sample();
        challenges.push(challenge);

        prev_folding_factors = on_challenge_received(
            &mut multilinears_f,
            &mut multilinears_ef,
            skip,
            &mut n_vars,
            &mut eq_factor,
            &mut sums,
            &mut missing_mul_factors,
            challenge,
            &ps,
            store_intermediate_foldings,
        );
        skip = 1;
        all_zerofiers = vec![false; sums.len()];
    }

    if let Some(prev_folding_factors) = prev_folding_factors {
        multilinears_f = multilinears_f.by_ref().fold(&prev_folding_factors).into();
        multilinears_ef = multilinears_ef.by_ref().fold(&prev_folding_factors).into();
    }

    (
        MultilinearPoint(challenges),
        multilinears_f.as_owned().unwrap(),
        multilinears_ef.as_owned().unwrap(),
        sums.into_iter().sum(),
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_and_send_polynomial<'a, EF, SC>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears_f: &mut MleGroup<'a, EF>,
    multilinears_ef: &mut MleGroup<'a, EF>,
    prev_folding_factors: Option<Vec<EF>>,
    computation: &SC,
    eq_factor: &Option<(Vec<EF>, MleOwned<EF>)>, // (a, b, c ...), eq_poly(b, c, ...)
    extra_data: &SC::ExtraData,
    alpha_powers: &[EF],
    all_zerofiers: &[bool],
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    sums: &[EF],
    missing_mul_factor: Option<EF>,
) -> Vec<DensePolynomial<EF>>
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<EF>,
{
    assert_eq!(all_zerofiers.len(), sums.len());
    let selectors = univariate_selectors::<PF<EF>>(skips);

    let mut all_p_evals = Vec::<Vec<(PF<EF>, EF)>>::new();

    let mut all_zs = Vec::new();
    for (degree, &is_zerofier) in computation.degrees().iter().zip(all_zerofiers) {
        let start = if is_zerofier {
            all_p_evals.push(
                (0..1 << skips)
                    .map(|i| (PF::<EF>::from_usize(i), EF::ZERO))
                    .collect(),
            );
            1 << skips
        } else {
            all_p_evals.push(vec![]);
            0
        };

        all_zs.push(
            (start..=degree * ((1 << skips) - 1))
                .filter(|&i| i != (1 << skips) - 1)
                .collect::<Vec<_>>(),
        );
    }

    let all_compute_folding_factors = all_zs
        .iter()
        .map(|zs| {
            zs.iter()
                .map(|&z| {
                    selectors
                        .iter()
                        .map(|s| s.evaluate(PF::<EF>::from_usize(z)))
                        .collect::<Vec<_>>()
                })
                .collect()
        })
        .collect::<Vec<Vec<Vec<PF<EF>>>>>();

    let sc_params = SumcheckComputeParams {
        skips,
        eq_mle: eq_factor.as_ref().map(|(_, eq_mle)| eq_mle),
        first_eq_factor: eq_factor
            .as_ref()
            .map(|(first_eq_factor, _)| first_eq_factor[0]),
        computation,
        extra_data,
        alpha_powers,
        missing_mul_factor,
        sums: &sums,
    };
    let all_computed_evals = match &prev_folding_factors {
        Some(prev_folding_factors) => {
            let (computed_p_evals, folded_multilinears_f, folded_multilinears_ef) =
                fold_and_sumcheck_compute_vec(
                    prev_folding_factors,
                    &multilinears_f.by_ref(),
                    &multilinears_ef.by_ref(),
                    sc_params,
                    &all_zs,
                    &all_compute_folding_factors,
                );
            *multilinears_f = folded_multilinears_f.into();
            *multilinears_ef = folded_multilinears_ef.into();
            computed_p_evals
        }
        None => sumcheck_compute_vec(
            &multilinears_f.by_ref(),
            &multilinears_ef.by_ref(),
            sc_params,
            &all_zs,
            &all_compute_folding_factors,
        ),
    };

    let mut all_pols = vec![];

    for (p_evals, (computed_evals, (is_zerofier, &sum))) in all_p_evals.iter_mut().zip(
        all_computed_evals
            .iter()
            .zip(all_zerofiers.iter().zip(sums)),
    ) {
        p_evals.extend(computed_evals);

        if !is_zerofier {
            let missing_sum_z = if let Some((eq_factor, _)) = eq_factor {
                (sum - (0..(1 << skips) - 1)
                    .map(|i| p_evals[i].1 * selectors[i].evaluate(eq_factor[0]))
                    .sum::<EF>())
                    / selectors[(1 << skips) - 1].evaluate(eq_factor[0])
            } else {
                sum - p_evals[..(1 << skips) - 1]
                    .iter()
                    .map(|(_, s)| *s)
                    .sum::<EF>()
            };
            p_evals.push((PF::<EF>::from_usize((1 << skips) - 1), missing_sum_z));
        }

        let mut p = DensePolynomial::lagrange_interpolation(&p_evals).unwrap();

        if let Some((eq_factor, _)) = &eq_factor {
            // https://eprint.iacr.org/2024/108.pdf Section 3.2
            // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
            p *= &DensePolynomial::lagrange_interpolation(
                &(0..1 << skips)
                    .into_iter()
                    .map(|i| (PF::<EF>::from_usize(i), selectors[i].evaluate(eq_factor[0])))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }

        // sanity check
        assert_eq!(
            (0..1 << skips)
                .map(|i| p.evaluate(EF::from_usize(i)))
                .sum::<EF>(),
            sum
        );

        all_pols.push(p);
    }

    prover_state
        .add_extension_scalars(&all_pols.iter().cloned().sum::<DensePolynomial<_>>().coeffs);

    all_pols
}

#[allow(clippy::too_many_arguments)]
fn on_challenge_received<'a, EF: ExtensionField<PF<EF>>>(
    multilinears_f: &mut MleGroup<'a, EF>,
    multilinears_ef: &mut MleGroup<'a, EF>,
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    n_vars: &mut usize,
    eq_factor: &mut Option<(Vec<EF>, MleOwned<EF>)>, // (a, b, c ...), eq_poly(b, c, ...)
    sums: &mut [EF],
    missing_mul_factor: &mut Option<EF>,
    challenge: EF,
    all_pols: &[DensePolynomial<EF>],
    store_intermediate_foldings: bool,
) -> Option<Vec<EF>> {
    sums.iter_mut().zip_eq(all_pols).for_each(|(sum, p)| {
        *sum = p.evaluate(challenge);
    });

    *n_vars -= skips;

    let selectors = univariate_selectors::<PF<EF>>(skips);

    if let Some((eq_factor, eq_mle)) = eq_factor {
        *missing_mul_factor = Some(
            selectors
                .iter()
                .map(|s| s.evaluate(eq_factor[0]) * s.evaluate(challenge))
                .sum::<EF>()
                * missing_mul_factor.unwrap_or(EF::ONE)
                / (EF::ONE - eq_factor.get(1).copied().unwrap_or_default()),
        );
        eq_factor.remove(0);
        eq_mle.truncate(eq_mle.by_ref().packed_len() / 2);
    }
    // return the folding_factors
    let selectors = selectors
        .iter()
        .map(|s| s.evaluate(challenge))
        .collect::<Vec<_>>();

    if store_intermediate_foldings {
        *multilinears_f = multilinears_f.by_ref().fold(&selectors).into();
        *multilinears_ef = multilinears_ef.by_ref().fold(&selectors).into();
        None
    } else {
        Some(selectors)
    }
}
