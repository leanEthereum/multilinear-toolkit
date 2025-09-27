use crate::*;
use fiat_shamir::*;
use p3_field::ExtensionField;

#[derive(Debug, Clone)]
pub enum Mle<'a, EF: ExtensionField<PF<EF>>> {
    Owned(MleOwned<EF>),
    Ref(MleRef<'a, EF>),
}

impl<'a, EF: ExtensionField<PF<EF>>> Mle<'a, EF> {
    pub fn by_ref(&'a self) -> MleRef<'a, EF> {
        match self {
            Self::Owned(owned) => owned.by_ref(),
            Self::Ref(r) => r.clone(),
        }
    }

    pub fn evaluate(&self, point: &MultilinearPoint<EF>) -> EF {
        self.by_ref().evaluate(point)
    }
}
