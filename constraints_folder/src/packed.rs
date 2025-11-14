use fiat_shamir::*;
use p3_air::AirBuilder;
use p3_field::ExtensionField;

#[derive(Debug)]
pub struct ConstraintFolderPackedBase<'a, EF: ExtensionField<PF<EF>>> {
    pub main: &'a [PFPacking<EF>],
    pub alpha_powers: &'a [EF],
    pub accumulator: EFPacking<EF>,
    pub constraint_index: usize,
}

impl<'a, EF: ExtensionField<PF<EF>>> AirBuilder for ConstraintFolderPackedBase<'a, EF> {
    type F = PFPacking<EF>;
    type Expr = PFPacking<EF>;
    type Var = PFPacking<EF>;

    #[inline]
    fn main(&self) -> &[PFPacking<EF>] {
        self.main
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        let x: PFPacking<EF> = x.into();
        self.accumulator += Into::<EFPacking<EF>>::into(alpha_power) * x;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _array: [I; N]) {
        unreachable!();
        // let expr_array = array.map(Into::into);
        // self.accumulator += EFPacking::<EF>::from_basis_coefficients_fn(|i| {
        //     let alpha_powers = &self.decomposed_alpha_powers[i]
        //         [self.constraint_index..(self.constraint_index + N)];
        //     PFPacking::<EF>::packed_linear_combination::<N>(alpha_powers, &expr_array)
        // });
        // self.constraint_index += N;
    }
}

#[derive(Debug)]
pub struct ConstraintFolderPackedExtension<'a, EF: ExtensionField<PF<EF>>> {
    pub main: &'a [EFPacking<EF>],
    pub alpha_powers: &'a [EF],
    pub accumulator: EFPacking<EF>,
    pub constraint_index: usize,
}

impl<'a, EF: ExtensionField<PF<EF>>> AirBuilder for ConstraintFolderPackedExtension<'a, EF> {
    type F = PFPacking<EF>;
    type Expr = EFPacking<EF>;
    type Var = EFPacking<EF>;

    #[inline]
    fn main(&self) -> &[EFPacking<EF>] {
        &self.main
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        let x: EFPacking<EF> = x.into();
        self.accumulator += x * alpha_power;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _array: [I; N]) {
        unreachable!();
    }
}
