use p3_field::{ExtensionField, Field};

pub fn apply_matrix<const WIDTH: usize, F: Field, EF: ExtensionField<F>>(
    matrix: &[[F; WIDTH]; WIDTH],
    input: &[EF],
) -> Vec<EF> {
    assert_eq!(input.len(), WIDTH);
    let mut output = EF::zero_vec(WIDTH);
    for i in 0..WIDTH {
        for j in 0..WIDTH {
            output[i] += input[j] * matrix[i][j];
        }
    }
    output
}

pub fn transpose_matrix<const WIDTH: usize, F: Field>(
    matrix: &[[F; WIDTH]; WIDTH],
) -> [[F; WIDTH]; WIDTH] {
    let mut transposed = [[F::ZERO; WIDTH]; WIDTH];
    for i in 0..WIDTH {
        for j in 0..WIDTH {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

pub fn inverse_matrix<const WIDTH: usize, F: Field>(
    matrix: &[[F; WIDTH]; WIDTH],
) -> [[F; WIDTH]; WIDTH] {
    // Create an augmented matrix [A | I]
    let mut augmented: Vec<Vec<F>> = vec![vec![F::ZERO; WIDTH * 2]; WIDTH];

    for i in 0..WIDTH {
        for j in 0..WIDTH {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][WIDTH + i] = F::ONE;
    }

    // Forward elimination with partial pivoting
    for col in 0..WIDTH {
        // Find pivot
        let mut pivot_row = col;
        for i in (col + 1)..WIDTH {
            if augmented[i][col] != F::ZERO {
                pivot_row = i;
                break;
            }
        }

        // Swap rows if needed
        if pivot_row != col {
            augmented.swap(col, pivot_row);
        }

        // Scale pivot row
        let pivot = augmented[col][col];
        assert!(pivot != F::ZERO, "Matrix is singular");

        let pivot_inv = pivot.inverse();
        for j in 0..(WIDTH * 2) {
            augmented[col][j] *= pivot_inv;
        }

        // Eliminate column in other rows
        for i in 0..WIDTH {
            if i != col {
                let factor = augmented[i][col];
                let col_row = augmented[col].clone();
                for j in 0..(WIDTH * 2) {
                    augmented[i][j] -= factor * col_row[j];
                }
            }
        }
    }

    // Extract the inverse matrix from the right side
    let mut inverse = [[F::ZERO; WIDTH]; WIDTH];
    for i in 0..WIDTH {
        for j in 0..WIDTH {
            inverse[i][j] = augmented[i][WIDTH + j];
        }
    }

    inverse
}
