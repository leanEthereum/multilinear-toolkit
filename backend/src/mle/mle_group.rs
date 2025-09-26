use crate::*;
use fiat_shamir::*;
use p3_field::ExtensionField;

#[derive(Debug)]
pub enum MleGroup<'a, EF: ExtensionField<PF<EF>>> {
    Owned(MleGroupOwned<EF>),
    Ref(MleGroupRef<'a, EF>),
}

impl<'a, EF: ExtensionField<PF<EF>>> From<MleGroupOwned<EF>> for MleGroup<'a, EF> {
    fn from(owned: MleGroupOwned<EF>) -> Self {
        MleGroup::Owned(owned)
    }
}

impl<'a, EF: ExtensionField<PF<EF>>> From<MleGroupRef<'a, EF>> for MleGroup<'a, EF> {
    fn from(r: MleGroupRef<'a, EF>) -> Self {
        MleGroup::Ref(r)
    }
}

impl<'a, EF: ExtensionField<PF<EF>>> MleGroup<'a, EF> {
    pub fn by_ref(&'a self) -> MleGroupRef<'a, EF> {
        match self {
            Self::Owned(owned) => owned.by_ref(),
            Self::Ref(r) => r.clone(),
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            Self::Owned(owned) => owned.n_vars(),
            Self::Ref(r) => r.n_vars(),
        }
    }

    pub const fn n_columns(&self) -> usize {
        match self {
            Self::Owned(owned) => owned.n_columns(),
            Self::Ref(r) => r.n_columns(),
        }
    }

    pub fn fold_in_large_field_in_place(&mut self, scalars: &[EF]) {
        match self {
            Self::Owned(owned) => owned.fold_in_large_field_in_place(scalars),
            Self::Ref(_) => {
                *self = self.by_ref().fold(scalars).into();
            }
        }
    }

    pub fn as_owned(self) -> Option<MleGroupOwned<EF>> {
        match self {
            Self::Owned(owned) => Some(owned),
            Self::Ref(_) => None,
        }
    }
}
