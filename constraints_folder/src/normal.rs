use fiat_shamir::*;
use p3_air::AirBuilder;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;

#[derive(Debug)]
pub struct ConstraintFolder<'a, NF, EF>
where
    NF: ExtensionField<PF<EF>>,
    EF: ExtensionField<NF>,
{
    pub main: (RowMajorMatrixView<'a, NF>, RowMajorMatrixView<'a, EF>),
    pub alpha_powers: &'a [EF],
    pub accumulator: EF,
    pub constraint_index: usize,
}

impl<'a, NF, EF> AirBuilder for ConstraintFolder<'a, NF, EF>
where
    NF: ExtensionField<PF<EF>>,
    EF: Field + ExtensionField<NF>,
{
    type F = PF<EF>;
    type Expr = NF;
    type Var = NF;
    type Var2 = EF;
    type M1 = RowMajorMatrixView<'a, NF>;
    type M2 = RowMajorMatrixView<'a, EF>;

    #[inline]
    fn main(&self) -> (Self::M1, Self::M2) {
        self.main
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: NF = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += alpha_power * x;
        self.constraint_index += 1;
    }

    fn assert_zero_2(&mut self, x: Self::Var2) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += alpha_power * x;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _: [I; N]) {
        todo!()
    }
}
