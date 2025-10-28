use backend::*;
use fiat_shamir::*;
use p3_field::ExtensionField;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::*;

pub fn verify_poseidon_gkr_sumcheck<EF>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    skips: usize,
    point: &[EF],
    claimed_cubes: Vec<EF>,
    claimed_transparent_evals: Vec<EF>,
) -> Result<(Vec<EF>, Vec<EF>, Vec<EF>), ProofError>
where
    EF: ExtensionField<PF<EF>>,
{
    let res = sumcheck_verify_with_univariate_skip_in_parallel(
        verifier_state,
        [
            vec![4; claimed_cubes.len()],
            vec![2; claimed_transparent_evals.len()],
        ]
        .concat(),
        point.len() + skips - 1,
        skips,
    )?;

    for ((v1, _), v2) in res
        .iter()
        .zip(claimed_cubes.iter().chain(claimed_transparent_evals.iter()))
    {
        if v1 != v2 {
            return Err(ProofError::InvalidProof);
        }
    }

    let next_cube_evals = verifier_state.next_extension_scalars_vec(claimed_cubes.len())?;
    let next_transparent_values =
        verifier_state.next_extension_scalars_vec(claimed_transparent_evals.len())?;

    let sc_point = res[0].1.point.0.clone();
    let eq_eval = eq_poly_with_skip(&sc_point, point, skips);
    for ((_, eval), inner_value) in res[..claimed_cubes.len()].into_iter().zip(&next_cube_evals) {
        if eq_eval * inner_value.cube() != eval.value {
            return Err(ProofError::InvalidProof);
        }
    }
    for ((_, eval), inner_value) in res[claimed_cubes.len()..]
        .into_iter()
        .zip(&next_transparent_values)
    {
        if eq_eval * *inner_value != eval.value {
            return Err(ProofError::InvalidProof);
        }
    }

    Ok((sc_point, next_cube_evals, next_transparent_values))
}

pub fn prove_poseidon_gkr_sumcheck<'a, EF: ExtensionField<PF<EF>>>(
    mut skip: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears_to_cube: Vec<&[PFPacking<EF>]>,
    multilinears_transparent: Vec<&[PFPacking<EF>]>,
    mut eq_point: Vec<EF>,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut claimed_cubes: Vec<EF>,
    mut claimed_transparent_evals: Vec<EF>,
) -> (Vec<EF>, Vec<EF>, Vec<EF>) {
    let mut multilinears = multilinears_to_cube
        .into_iter()
        .chain(multilinears_transparent.into_iter())
        .map(|m| Mle::Ref(MleRef::BasePacked(m)))
        .collect::<Vec<_>>();
    let n_rounds = multilinears[0].by_ref().n_vars() - skip + 1;

    let mut eq_mle = MleOwned::ExtensionPacked(eval_eq_packed(&eq_point[1..]));
    let mut n_vars = multilinears[0].by_ref().n_vars();

    let mut challenges = Vec::new();
    let mut missing_mul_factors = None;
    let mut prev_folding_factors = None;
    for _ in 0..n_rounds {
        // If Packing is enabled, and there are too little variables, we unpack everything:
        if multilinears[0].by_ref().is_packed() && n_vars <= 1 + packing_log_width::<EF>() {
            // unpack
            // multilinears = multilinears.by_ref().unpack().into();
            multilinears = multilinears
                .into_iter()
                .map(|m| Mle::Owned(m.by_ref().unpack().as_owned().unwrap()))
                .collect();
            eq_mle = MleOwned::Extension(unpack_extension(
                eq_mle.by_ref().as_extension_packed().unwrap(),
            ));
        }

        let ps = compute_and_send_poseidon_gkr_polynomials(
            skip,
            &mut multilinears,
            prev_folding_factors,
            (&eq_point, &eq_mle),
            prover_state,
            &claimed_cubes,
            &claimed_transparent_evals,
            missing_mul_factors,
        );
        let challenge = prover_state.sample();
        challenges.push(challenge);

        prev_folding_factors = on_challenge_received(
            skip,
            &mut n_vars,
            (&mut eq_point, &mut eq_mle),
            &mut claimed_cubes,
            &mut claimed_transparent_evals,
            &mut missing_mul_factors,
            challenge,
            &ps,
        );
        skip = 1;
    }

    let prev_folding_factors = prev_folding_factors.unwrap();

    let inner_evals = multilinears
        .iter()
        .map(|m| {
            m.by_ref()
                .as_group()
                .fold(&prev_folding_factors)
                .as_single()
                .as_extension()
                .unwrap()
                .as_constant()
        })
        .collect::<Vec<_>>();

    let next_cube_evals = inner_evals[..claimed_cubes.len()].to_vec();
    let next_transparent_values = inner_evals[claimed_cubes.len()..].to_vec();
    prover_state.add_extension_scalars(&next_cube_evals);
    prover_state.add_extension_scalars(&next_transparent_values);
    (challenges, next_cube_evals, next_transparent_values)
}

fn compute_and_send_poseidon_gkr_polynomials<'a, EF: ExtensionField<PF<EF>>>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &mut Vec<Mle<'a, EF>>,
    prev_folding_factors: Option<Vec<EF>>,
    eq_factor: (&[EF], &MleOwned<EF>), // (a, b, c ...), eq_poly(b, c, ...)
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    claimed_cubes: &[EF],
    claimed_transparent_evals: &[EF],
    missing_mul_factor: Option<EF>,
) -> Vec<DensePolynomial<EF>> {
    let selectors = univariate_selectors::<PF<EF>>(skips);

    let zs_cubes = (0..=3 * ((1 << skips) - 1))
        .filter(|&i| i != (1 << skips) - 1)
        .collect::<Vec<_>>();
    let zs_transparent = (0..=((1 << skips) - 1))
        .filter(|&i| i != (1 << skips) - 1)
        .collect::<Vec<_>>();

    let compute_folding_factors_cubes = zs_cubes
        .iter()
        .map(|&z| {
            selectors
                .iter()
                .map(|s| s.evaluate(PF::<EF>::from_usize(z)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<PF<EF>>>>();

    let compute_folding_factors_transparent = zs_transparent
        .iter()
        .map(|&z| {
            selectors
                .iter()
                .map(|s| s.evaluate(PF::<EF>::from_usize(z)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<PF<EF>>>>();

    let sc_params_cube = (0..claimed_cubes.len())
        .map(|i| SumcheckComputeParams {
            skips,
            eq_mle: Some(&eq_factor.1),
            first_eq_factor: Some(eq_factor.0[0]),
            folding_factors: &compute_folding_factors_cubes,
            computation: &CubeComputation {},
            batching_scalars: &[],
            missing_mul_factor,
            sum: claimed_cubes[i],
        })
        .collect::<Vec<_>>();

    let sc_params_transparent = (0..claimed_transparent_evals.len())
        .map(|i| SumcheckComputeParams {
            skips,
            eq_mle: Some(&eq_factor.1),
            first_eq_factor: Some(eq_factor.0[0]),
            folding_factors: &compute_folding_factors_transparent,
            computation: &TransparentComputation {},
            batching_scalars: &[],
            missing_mul_factor,
            sum: claimed_transparent_evals[i],
        })
        .collect::<Vec<_>>();

    let (cube_multilinears, transparent_multilinears) =
        multilinears.split_at_mut(claimed_cubes.len());
    let mut p_evals: Vec<Vec<(PF<EF>, EF)>> = cube_multilinears
        .par_iter_mut()
        .zip(sc_params_cube)
        .map(|(poly, sc_param)| match &prev_folding_factors {
            Some(prev_folding_factors) => {
                let (computed_p_evals, folded_multilinears) = fold_and_sumcheck_compute(
                    prev_folding_factors,
                    &poly.by_ref().as_group(),
                    sc_param,
                    &zs_cubes,
                );
                *poly = Mle::Owned(folded_multilinears.as_single());
                computed_p_evals
            }
            None => sumcheck_compute(&poly.by_ref().as_group(), sc_param, &zs_cubes),
        })
        .chain(
            transparent_multilinears
                .par_iter_mut()
                .zip(sc_params_transparent)
                .map(|(poly, sc_param)| match &prev_folding_factors {
                    Some(prev_folding_factors) => {
                        let (computed_p_evals, folded_multilinears) = fold_and_sumcheck_compute(
                            prev_folding_factors,
                            &poly.by_ref().as_group(),
                            sc_param,
                            &zs_transparent,
                        );
                        *poly = Mle::Owned(folded_multilinears.as_single());
                        computed_p_evals
                    }
                    None => sumcheck_compute(&poly.by_ref().as_group(), sc_param, &zs_transparent),
                }),
        )
        .collect::<Vec<_>>();

    let mut inerpolated = vec![];
    for (evals, &sum) in p_evals
        .iter_mut()
        .zip(claimed_cubes.iter().chain(claimed_transparent_evals.iter()))
    {
        let missing_sum_z = {
            (sum - (0..(1 << skips) - 1)
                .map(|i| evals[i].1 * selectors[i].evaluate(eq_factor.0[0]))
                .sum::<EF>())
                / selectors[(1 << skips) - 1].evaluate(eq_factor.0[0])
        };

        evals.push((PF::<EF>::from_usize((1 << skips) - 1), missing_sum_z));

        let mut p = DensePolynomial::lagrange_interpolation(&evals).unwrap();

        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= &DensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| {
                    (
                        PF::<EF>::from_usize(i),
                        selectors[i].evaluate(eq_factor.0[0]),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

        // sanity check
        assert_eq!(
            (0..1 << skips)
                .map(|i| p.evaluate(EF::from_usize(i)))
                .sum::<EF>(),
            sum
        );

        prover_state.add_extension_scalars(&p.coeffs);

        inerpolated.push(p);
    }

    inerpolated
}

#[allow(clippy::too_many_arguments)]
fn on_challenge_received<'a, EF: ExtensionField<PF<EF>>>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    n_vars: &mut usize,
    eq_factor: (&mut Vec<EF>, &mut MleOwned<EF>), // (a, b, c ...), eq_poly(b, c, ...)
    claimed_cubes: &mut Vec<EF>,
    claimed_transparent_evals: &mut Vec<EF>,
    missing_mul_factor: &mut Option<EF>,
    challenge: EF,
    p: &[DensePolynomial<EF>],
) -> Option<Vec<EF>> {
    *claimed_cubes = p[..claimed_cubes.len()]
        .iter()
        .map(|poly| poly.evaluate(challenge))
        .collect::<Vec<_>>();
    *claimed_transparent_evals = p[claimed_cubes.len()..]
        .iter()
        .map(|poly| poly.evaluate(challenge))
        .collect::<Vec<_>>();
    *n_vars -= skips;

    let selectors = univariate_selectors::<PF<EF>>(skips);

    *missing_mul_factor = Some(
        selectors
            .iter()
            .map(|s| s.evaluate(eq_factor.0[0]) * s.evaluate(challenge))
            .sum::<EF>()
            * missing_mul_factor.unwrap_or(EF::ONE)
            / (EF::ONE - eq_factor.0.get(1).copied().unwrap_or_default()),
    );
    eq_factor.0.remove(0);
    eq_factor.1.truncate(eq_factor.1.by_ref().packed_len() / 2);

    // return the folding_factors
    let selectors = selectors
        .iter()
        .map(|s| s.evaluate(challenge))
        .collect::<Vec<_>>();

    Some(selectors)
}

use crate::{SumcheckComputation, SumcheckComputationPacked};

#[derive(Debug)]
struct CubeComputation;

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

#[derive(Debug)]
struct TransparentComputation;

impl<IF: ExtensionField<PF<EF>>, EF: ExtensionField<IF>> SumcheckComputation<IF, EF>
    for TransparentComputation
{
    fn eval(&self, point: &[IF], _: &[EF]) -> EF {
        // TODO avoid embedding overhead
        EF::from(point[0])
    }
    fn degree(&self) -> usize {
        1
    }
}

impl<EF: ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for TransparentComputation {
    fn eval_packed_base(&self, point: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        // TODO avoid embedding overhead
        EFPacking::<EF>::from(point[0])
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        point[0]
    }
    fn degree(&self) -> usize {
        1
    }
}
