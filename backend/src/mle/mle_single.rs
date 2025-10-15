use crate::*;
use fiat_shamir::*;
use p3_field::ExtensionField;

#[derive(Debug, Clone)]
pub enum Mle<'a, EF: ExtensionField<PF<EF>>> {
    Owned(MleOwned<EF>),
    Ref(MleRef<'a, EF>),
}

impl<EF: ExtensionField<PF<EF>>> From<MleOwned<EF>> for Mle<'_, EF> {
    fn from(value: MleOwned<EF>) -> Self {
        Self::Owned(value)
    }
}

impl<'a, EF: ExtensionField<PF<EF>>> From<MleRef<'a, EF>> for Mle<'a, EF> {
    fn from(value: MleRef<'a, EF>) -> Self {
        Self::Ref(value)
    }
}

impl<'a, EF: ExtensionField<PF<EF>>> Mle<'a, EF> {
    pub fn by_ref(&'a self) -> MleRef<'a, EF> {
        match self {
            Self::Owned(owned) => owned.by_ref(),
            Self::Ref(r) => r.clone(),
        }
    }

    pub fn as_owned_mut(&mut self) -> Option<&mut MleOwned<EF>> {
        match self {
            Self::Owned(o) => Some(o),
            _ => None,
        }
    }

    pub fn as_owned(self) -> Option<MleOwned<EF>> {
        match self {
            Self::Owned(o) => Some(o),
            _ => None,
        }
    }

    pub fn evaluate(&self, point: &MultilinearPoint<EF>) -> EF {
        self.by_ref().evaluate(point)
    }

    pub fn pack(&'a self) -> Self {
        match self {
            Self::Owned(poly) => poly.pack(),
            Self::Ref(poly) => poly.pack(),
        }
    }
}
