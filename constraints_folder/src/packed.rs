use fiat_shamir::*;
use p3_air::AirBuilder;
use p3_field::ExtensionField;
use p3_matrix::dense::RowMajorMatrixView;

#[derive(Debug)]
pub struct ConstraintFolderPackedBase<'a, EF: ExtensionField<PF<EF>>> {
    pub main: (
        RowMajorMatrixView<'a, PFPacking<EF>>,
        RowMajorMatrixView<'a, EFPacking<EF>>,
    ),
    pub alpha_powers: &'a [EF],
    pub accumulator: EFPacking<EF>,
    pub constraint_index: usize,
}

impl<'a, EF: ExtensionField<PF<EF>>> AirBuilder for ConstraintFolderPackedBase<'a, EF> {
    type F = PFPacking<EF>;
    type Expr = PFPacking<EF>;
    type Var = PFPacking<EF>;
    type Var2 = EFPacking<EF>;
    type M1 = RowMajorMatrixView<'a, PFPacking<EF>>;
    type M2 = RowMajorMatrixView<'a, EFPacking<EF>>;

    #[inline]
    fn main(&self) -> (Self::M1, Self::M2) {
        self.main
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        let x: PFPacking<EF> = x.into();
        self.accumulator += Into::<EFPacking<EF>>::into(alpha_power) * x;
        self.constraint_index += 1;
    }

    fn assert_zero_2(&mut self, x: Self::Var2) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += x * alpha_power;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _array: [I; N]) {
        todo!();
    }
}

#[derive(Debug)]
pub struct ConstraintFolderPackedExtension<'a, EF: ExtensionField<PF<EF>>> {
    pub main: (
        RowMajorMatrixView<'a, EFPacking<EF>>,
        RowMajorMatrixView<'a, EFPacking<EF>>,
    ),
    pub alpha_powers: &'a [EF],
    pub accumulator: EFPacking<EF>,
    pub constraint_index: usize,
}

impl<'a, EF: ExtensionField<PF<EF>>> AirBuilder for ConstraintFolderPackedExtension<'a, EF> {
    type F = PFPacking<EF>;
    type Expr = EFPacking<EF>;
    type Var = EFPacking<EF>;
    type Var2 = EFPacking<EF>;
    type M1 = RowMajorMatrixView<'a, EFPacking<EF>>;
    type M2 = RowMajorMatrixView<'a, EFPacking<EF>>;

    #[inline]
    fn main(&self) -> (Self::M1, Self::M2) {
        self.main
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        let x: EFPacking<EF> = x.into();
        self.accumulator += x * alpha_power;
        self.constraint_index += 1;
    }

    fn assert_zero_2(&mut self, x: Self::Var2) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += x * alpha_power;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _array: [I; N]) {
        todo!();
    }
}
