use backend::*;
use fiat_shamir::*;
use p3_field::ExtensionField;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::*;

#[allow(clippy::too_many_arguments)]
pub fn sumcheck_prove<'a, EF, SC, SCP, M: Into<MleGroup<'a, EF>>>(
    skip: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: M,
    computation: &SC,
    computation_packed: &SCP,
    batching_scalars: &[EF],
    eq_factor: Option<(Vec<EF>, Option<MleOwned<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    sum: EF,
    missing_mul_factors: Option<EF>,
) -> (MultilinearPoint<EF>, Vec<EF>, EF)
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<PF<EF>, EF> + SumcheckComputation<EF, EF> + 'static,
    SCP: SumcheckComputationPacked<EF>,
{
    let multilinears: MleGroup<'a, EF> = multilinears.into();
    let n_rounds = multilinears.by_ref().n_vars() - skip + 1;
    let (challenges, final_folds, final_sum) = sumcheck_prove_many_rounds(
        skip,
        multilinears,
        computation,
        computation_packed,
        batching_scalars,
        eq_factor,
        is_zerofier,
        prover_state,
        sum,
        missing_mul_factors,
        n_rounds,
    );

    let final_folds = final_folds
        .by_ref()
        .as_extension()
        .unwrap()
        .iter()
        .map(|m| {
            assert_eq!(m.len(), 1);
            m[0]
        })
        .collect::<Vec<_>>();

    (challenges, final_folds, final_sum)
}
#[allow(clippy::too_many_arguments)]
pub fn sumcheck_prove_many_rounds<'a, EF, SC, SCP, M: Into<MleGroup<'a, EF>>>(
    mut skip: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: M,
    computation: &SC,
    computation_packed: &SCP,
    batching_scalars: &[EF],
    mut eq_factor: Option<(Vec<EF>, Option<MleOwned<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    mut is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut sum: EF,
    mut missing_mul_factors: Option<EF>,
    n_rounds: usize,
) -> (MultilinearPoint<EF>, MleGroup<'a, EF>, EF)
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<PF<EF>, EF> + SumcheckComputation<EF, EF> + 'static,
    SCP: SumcheckComputationPacked<EF>,
{
    let mut multilinears: MleGroup<'a, EF> = multilinears.into();
    let mut eq_factor: Option<(Vec<EF>, MleOwned<EF>)> = eq_factor.take().map(|(eq_point, eq_mle)| {
        let eq_mle = eq_mle.unwrap_or_else(|| {
            let eval_eq_ext = eval_eq(&eq_point[1..]);
            if multilinears.by_ref().is_packed() {
                MleOwned::ExtensionPacked(pack_extension(&eval_eq_ext))
            } else {
                MleOwned::Extension(eval_eq_ext)
            }
        });
        (eq_point, eq_mle)
    });
    let mut n_vars = multilinears.by_ref().n_vars();
    if let Some((eq_point, eq_mle)) = &eq_factor {
        assert_eq!(eq_point.len(), n_vars - skip + 1);
        assert_eq!(eq_mle.by_ref().n_vars(), eq_point.len() - 1);
        assert_eq!(eq_mle.by_ref().is_packed(), multilinears.by_ref().is_packed());
    }

    let mut challenges = Vec::new();
    for _ in 0..n_rounds {
        // If Packing is enabled, and there are too little variables, we unpack everything:
        if multilinears.by_ref().is_packed() && n_vars <= 1 + packing_log_width::<EF>() {
            // unpack
            multilinears = multilinears.by_ref().unpack().into();
            if let Some((_, eq_mle)) = &mut eq_factor {
                *eq_mle = MleOwned::Extension(unpack_extension(eq_mle.by_ref().as_extension_packed().unwrap()));
            }
        }

        let ps = compute_and_send_polynomial(
            skip,
            &multilinears,
            computation,
            computation_packed,
            &eq_factor,
            batching_scalars,
            is_zerofier,
            prover_state,
            sum,
            missing_mul_factors,
        );
        let challenge = prover_state.sample();
        challenges.push(challenge);

        on_challenge_received(
            skip,
            &mut multilinears,
            &mut n_vars,
            &mut eq_factor,
            &mut sum,
            &mut missing_mul_factors,
            challenge,
            &ps,
        );
        skip = 1;
        is_zerofier = false;
    }

    (MultilinearPoint(challenges), multilinears, sum)
}

#[allow(clippy::too_many_arguments)]
fn compute_and_send_polynomial<'a, EF, SC, SCP>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &MleGroup<'a, EF>,
    computation: &SC,
    computations_packed: &SCP,
    eq_factor: &Option<(Vec<EF>, MleOwned<EF>)>, // (a, b, c ...), eq_poly(b, c, ...)
    batching_scalars: &[EF],
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    sum: EF,
    missing_mul_factor: Option<EF>,
) -> DensePolynomial<EF>
where
    EF: ExtensionField<PF<EF>>,
    SC: SumcheckComputation<PF<EF>, EF> + SumcheckComputation<EF, EF> + 'static,
    SCP: SumcheckComputationPacked<EF>,
{
    let selectors = univariate_selectors::<PF<EF>>(skips);

    let mut p_evals = Vec::<(PF<EF>, EF)>::new();
    let start = if is_zerofier {
        p_evals.extend((0..1 << skips).map(|i| (PF::<EF>::from_usize(i), EF::ZERO)));
        1 << skips
    } else {
        0
    };

    let computation_degree = SumcheckComputation::<EF, EF>::degree(computation);
    let zs = (start..=computation_degree * ((1 << skips) - 1))
        .filter(|&i| i != (1 << skips) - 1)
        .collect::<Vec<_>>();

    let folding_scalars = zs
        .iter()
        .map(|&z| {
            selectors
                .iter()
                .map(|s| s.evaluate(PF::<EF>::from_usize(z)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<PF<EF>>>>();

    p_evals.extend(sumcheck_compute(
        &multilinears.by_ref(),
        SumcheckComputeParams {
            zs: &zs,
            skips,
            eq_mle: eq_factor.as_ref().map(|(_, eq_mle)| eq_mle),
            folding_scalars: &folding_scalars,
            computation,
            computation_packed: computations_packed,
            batching_scalars,
            missing_mul_factor,
            sum,
        },
    ));

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
                .into_par_iter()
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

    prover_state.add_extension_scalars(&p.coeffs);

    p
}

#[allow(clippy::too_many_arguments)]
fn on_challenge_received<'a, EF>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &mut MleGroup<'a, EF>,
    n_vars: &mut usize,
    eq_factor: &mut Option<(Vec<EF>, MleOwned<EF>)>, // (a, b, c ...), eq_poly(b, c, ...)
    sum: &mut EF,
    missing_mul_factor: &mut Option<EF>,
    challenge: EF,
    p: &DensePolynomial<EF>,
) where
    EF: ExtensionField<PF<EF>>,
{
    *sum = p.evaluate(challenge);
    *n_vars -= skips;

    let selectors = univariate_selectors::<PF<EF>>(skips);

    let folding_scalars = selectors
        .iter()
        .map(|s| s.evaluate(challenge))
        .collect::<Vec<_>>();
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

    multilinears.fold_in_large_field_in_place(&folding_scalars);
}
