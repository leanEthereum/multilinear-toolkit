use backend::*;
use fiat_shamir::*;
use p3_field::*;

pub fn sumcheck_verify<EF>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    n_vars: usize,
    degree: usize,
) -> Result<(EF, Evaluation<EF>), ProofError>
where
    EF: ExtensionField<PF<EF>>,
{
    let sumation_sets = vec![vec![EF::ZERO, EF::ONE]; n_vars];
    let max_degree_per_vars = vec![vec![degree; n_vars]];
    Ok(verify_core(verifier_state, max_degree_per_vars, sumation_sets)?[0].clone())
}

pub fn sumcheck_verify_with_custom_degree_at_first_round<EF>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    n_vars: usize,
    intial_degree: usize,
    remaining_degree: usize,
) -> Result<(EF, Evaluation<EF>), ProofError>
where
    EF: ExtensionField<PF<EF>>,
{
    let sumation_sets = vec![vec![EF::ZERO, EF::ONE]; n_vars];
    let mut max_degree_per_vars = vec![intial_degree; 1];
    max_degree_per_vars.extend(vec![remaining_degree; n_vars - 1]);
    Ok(verify_core(verifier_state, vec![max_degree_per_vars], sumation_sets)?[0].clone())
}

pub fn sumcheck_verify_with_univariate_skip<EF>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    degree: usize,
    n_vars: usize,
    skips: usize,
) -> Result<(EF, Evaluation<EF>), ProofError>
where
    EF: ExtensionField<PF<EF>>,
{
    Ok(sumcheck_verify_with_univariate_skip_in_parallel(
        verifier_state,
        vec![degree],
        n_vars,
        skips,
    )?[0]
        .clone())
}

pub fn sumcheck_verify_with_univariate_skip_in_parallel<EF>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    degrees: Vec<usize>,
    n_vars: usize,
    skips: usize,
) -> Result<Vec<(EF, Evaluation<EF>)>, ProofError>
where
    EF: ExtensionField<PF<EF>>,
{
    let mut all_max_degree_per_vars = Vec::new();
    for degree in degrees {
        let mut max_degree_per_vars = vec![degree * ((1 << skips) - 1)];
        max_degree_per_vars.extend(vec![degree; n_vars - skips]);
        all_max_degree_per_vars.push(max_degree_per_vars);
    }
    let mut sumation_sets = vec![(0..1 << skips).map(EF::from_usize).collect::<Vec<_>>()];
    sumation_sets.extend(vec![vec![EF::ZERO, EF::ONE]; n_vars - skips]);
    verify_core(verifier_state, all_max_degree_per_vars, sumation_sets)
}

fn verify_core<EF>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    max_degree_per_vars: Vec<Vec<usize>>,
    sumation_sets: Vec<Vec<EF>>,
) -> Result<Vec<(EF, Evaluation<EF>)>, ProofError>
where
    EF: ExtensionField<PF<EF>>,
{
    let n_sumchecks = max_degree_per_vars.len();
    max_degree_per_vars.iter().for_each(|deg_vec| {
        assert_eq!(deg_vec.len(), sumation_sets.len());
    });

    let mut challenges = Vec::new();
    let mut first_round = true;
    let mut sums = vec![EF::ZERO; n_sumchecks];
    let mut targets = vec![EF::ZERO; n_sumchecks];

    let n_vars = max_degree_per_vars[0].len();

    for var in 0..n_vars {
        let sumation_set = &sumation_sets[var];
        let mut pols = Vec::new();
        for i in 0..n_sumchecks {
            let deg = max_degree_per_vars[i][var];
            let coeffs = verifier_state.next_extension_scalars_vec(deg + 1)?;
            let pol = DensePolynomial::new(coeffs);
            let computed_sum = sumation_set.iter().map(|&s| pol.evaluate(s)).sum();
            if first_round {
                sums[i] = computed_sum;
            } else if targets[i] != computed_sum {
                return Err(ProofError::InvalidProof);
            }
            pols.push(pol);
        }
        first_round = false;

        let challenge = verifier_state.sample();
        challenges.push(challenge);
        for i in 0..n_sumchecks {
            targets[i] = pols[i].evaluate(challenge);
        }
    }

    Ok(sums
        .into_iter()
        .zip(targets.into_iter())
        .map(|(s, t)| (s, Evaluation::new(challenges.clone(), t)))
        .collect::<Vec<_>>())
}
