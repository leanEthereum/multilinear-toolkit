use crate::*;
use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use rayon::{join, prelude::*};
use std::borrow::Borrow;

pub trait EvaluationsList<F: Field> {
    fn num_variables(&self) -> usize;
    fn num_evals(&self) -> usize;
    fn evaluate<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF;
    fn as_constant(&self) -> F;
    fn evaluate_sparse<EF: ExtensionField<F>>(&self, points: &MultilinearPoint<EF>) -> EF;
}

impl<F: Field, EL: Borrow<[F]>> EvaluationsList<F> for EL {
    fn num_variables(&self) -> usize {
        self.borrow().len().ilog2() as usize
    }

    fn num_evals(&self) -> usize {
        self.borrow().len()
    }

    fn evaluate<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        eval_multilinear(self.borrow(), point)
    }

    fn as_constant(&self) -> F {
        assert_eq!(self.borrow().len(), 1);
        self.borrow()[0]
    }

    fn evaluate_sparse<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        assert_eq!(point.len(), self.num_variables());
        if point.is_empty() {
            return self.as_constant().into();
        }

        let initial_booleans = point
            .iter()
            .take_while(|&&x| x == EF::ZERO || x == EF::ONE)
            .map(|&x| if x == EF::ZERO { 0 } else { 1 })
            .collect::<Vec<_>>();

        if initial_booleans.len() != point.len()
            && [EF::ZERO, EF::ONE].contains(&point.last().unwrap())
        {
            tracing::warn!(
                "TODO, evaluate_sparse has not yet been optimized when booleans not at the start"
            );
        }

        let offset = initial_booleans.iter().fold(0, |acc, b| (acc << 1) | b);

        (&self.borrow()[offset << (point.len() - initial_booleans.len())
            ..((offset + 1) << (point.len() - initial_booleans.len()))])
            .evaluate(&MultilinearPoint(point[initial_booleans.len()..].to_vec()))
    }
}

/// Multiply the polynomial by a scalar factor.
#[must_use]
pub fn scale_poly<F: Field, EF: ExtensionField<F>>(poly: &[F], factor: EF) -> Vec<EF> {
    poly.par_iter().map(|&e| factor * e).collect()
}

fn eval_multilinear<F, EF>(evals: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the number of evaluations matches the number of variables in the point.
    //
    // This is a critical invariant: `evals.len()` must be exactly `2^point.len()`.
    debug_assert_eq!(evals.len(), 1 << point.len());

    // Select the optimal evaluation strategy based on the number of variables.
    match point {
        // Case: 0 Variables (Constant Polynomial)
        //
        // A polynomial with zero variables is just a constant.
        [] => evals[0].into(),

        // Case: 1 Variable (Linear Interpolation)
        //
        // This is the base case for the recursion: f(x) = f(0) * (1-x) + f(1) * x.
        // The expression is an optimized form: f(0) + x * (f(1) - f(0)).
        [x] => *x * (evals[1] - evals[0]) + evals[0],

        // Case: 2 Variables (Bilinear Interpolation)
        //
        // This is a fully unrolled version for 2 variables, avoiding recursive calls.
        [x0, x1] => {
            // Interpolate along the x1-axis for x0=0 to get `a0`.
            let a0 = *x1 * (evals[1] - evals[0]) + evals[0];
            // Interpolate along the x1-axis for x0=1 to get `a1`.
            let a1 = *x1 * (evals[3] - evals[2]) + evals[2];
            // Finally, interpolate between `a0` and `a1` along the x0-axis.
            a0 + (a1 - a0) * *x0
        }

        // Cases: 3 and 4 Variables
        //
        // These are further unrolled versions for 3 and 4 variables for maximum speed.
        // The logic is the same as the 2-variable case, just with more steps.
        [x0, x1, x2] => {
            let a00 = *x2 * (evals[1] - evals[0]) + evals[0];
            let a01 = *x2 * (evals[3] - evals[2]) + evals[2];
            let a10 = *x2 * (evals[5] - evals[4]) + evals[4];
            let a11 = *x2 * (evals[7] - evals[6]) + evals[6];
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = *x3 * (evals[1] - evals[0]) + evals[0];
            let a001 = *x3 * (evals[3] - evals[2]) + evals[2];
            let a010 = *x3 * (evals[5] - evals[4]) + evals[4];
            let a011 = *x3 * (evals[7] - evals[6]) + evals[6];
            let a100 = *x3 * (evals[9] - evals[8]) + evals[8];
            let a101 = *x3 * (evals[11] - evals[10]) + evals[10];
            let a110 = *x3 * (evals[13] - evals[12]) + evals[12];
            let a111 = *x3 * (evals[15] - evals[14]) + evals[14];
            let a00 = a000 + *x2 * (a001 - a000);
            let a01 = a010 + *x2 * (a011 - a010);
            let a10 = a100 + *x2 * (a101 - a100);
            let a11 = a110 + *x2 * (a111 - a110);
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }

        // General Case (5+ Variables)
        //
        // This handles all other cases, using one of two different strategies.
        [x, tail @ ..] => {
            // For a very large number of variables, the recursive approach is not the most efficient.
            //
            // We switch to a more direct, non-recursive algorithm that is better suited for wide parallelization.
            if point.len() >= 20 {
                // The `evals` are ordered lexicographically, meaning the first variable's bit changes the slowest.
                //
                // To align our computation with this memory layout, we process the point's coordinates in reverse.
                let mut point_rev = point.to_vec();
                point_rev.reverse();

                // Split the reversed point's coordinates into two halves:
                // - `z0` (low-order vars)
                // - `z1` (high-order vars).
                let mid = point_rev.len() / 2;
                let (z0, z1) = point_rev.split_at(mid);

                // Precomputation of Basis Polynomials
                //
                // The basis polynomial eq(v, p) can be split: eq(v, p) = eq(v_low, p_low) * eq(v_high, p_high).
                //
                // We precompute all `2^|z0|` values of eq(v_low, p_low) and store them in `left`.
                // We precompute all `2^|z1|` values of eq(v_high, p_high) and store them in `right`.

                // Allocate uninitialized memory for the low-order basis polynomial evaluations.
                let mut left = unsafe { uninitialized_vec(1 << z0.len()) };
                // Allocate uninitialized memory for the high-order basis polynomial evaluations.
                let mut right = unsafe { uninitialized_vec(1 << z1.len()) };

                // The `eval_eq` function requires the variables in their original order, so we reverse the halves back.
                let mut z0_ordered = z0.to_vec();
                z0_ordered.reverse();
                // Compute all eq(v_low, p_low) values and fill the `left` vector.
                compute_eval_eq::<_, _, false>(&z0_ordered, &mut left, EF::ONE);

                // Repeat the process for the high-order variables.
                let mut z1_ordered = z1.to_vec();
                z1_ordered.reverse();
                // Compute all eq(v_high, p_high) values and fill the `right` vector.
                compute_eval_eq::<_, _, false>(&z1_ordered, &mut right, EF::ONE);

                // Parallelized Final Summation
                //
                // This chain of operations computes the regrouped sum:
                // Σ_{v_high} eq(v_high, p_high) * (Σ_{v_low} f(v_high, v_low) * eq(v_low, p_low))
                evals
                    .par_chunks(left.len())
                    .zip_eq(right.par_iter())
                    .map(|(part, &c)| {
                        // This is the inner sum: a dot product between the evaluation chunk and the `left` basis values.
                        part.iter()
                            .zip_eq(left.iter())
                            .map(|(&a, &b)| b * a)
                            .sum::<EF>()
                            * c
                    })
                    .sum()
            } else {
                // For moderately sized inputs (5 to 19 variables), use the recursive strategy.
                //
                // Split the evaluations into two halves, corresponding to the first variable being 0 or 1.
                let (f0, f1) = evals.split_at(evals.len() / 2);

                // Recursively evaluate on the two smaller hypercubes.
                let (f0_eval, f1_eval) = {
                    // Only spawn parallel tasks if the subproblem is large enough to overcome
                    // the overhead of threading.
                    let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
                    if evals.len() > work_size {
                        join(|| eval_multilinear(f0, tail), || eval_multilinear(f1, tail))
                    } else {
                        // For smaller subproblems, execute sequentially.
                        (eval_multilinear(f0, tail), eval_multilinear(f1, tail))
                    }
                };
                // Perform the final linear interpolation for the first variable `x`.
                f0_eval + (f1_eval - f0_eval) * *x
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::{KoalaBear, QuinticExtensionFieldKB};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    type F = KoalaBear;
    type EF = QuinticExtensionFieldKB;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn test_evaluate_sparse() {
        let n_vars = 10;
        let mut rng = StdRng::seed_from_u64(0);
        let poly = (0..(1 << n_vars)).map(|_| rng.random()).collect::<Vec<F>>();
        for n_initial_booleans in 0..n_vars {
            for rep in 0..1 << n_initial_booleans {
                let mut point = (0..n_initial_booleans)
                    .map(|i| EF::from_usize((rep >> i) & 1))
                    .collect::<Vec<_>>();
                for _ in n_initial_booleans..n_vars {
                    point.push(rng.random());
                }
                assert_eq!(
                    poly.evaluate_sparse(&MultilinearPoint(point.clone())),
                    poly.evaluate(&MultilinearPoint(point))
                );
            }
        }
    }
}
