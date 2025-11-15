use fiat_shamir::*;
use p3_air::AirBuilder;
use p3_field::{ExtensionField, Field};

#[derive(Debug)]
pub struct ConstraintFolder<'a, NF, EF>
where
    NF: ExtensionField<PF<EF>>,
    EF: ExtensionField<NF>,
{
    pub main: &'a [NF],
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
    type FinalOutput = EF;

    #[inline]
    fn main(&self) -> &[NF] {
        self.main
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: NF = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += alpha_power * x;
        self.constraint_index += 1;
    }

    #[inline]
    fn add_custom(&mut self, value: Self::FinalOutput) {
        self.accumulator += value;
        self.constraint_index += 1;
    }
}
