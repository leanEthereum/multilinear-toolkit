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

    pub fn fold_in_place(&mut self, weights: &[EF]) {
        match self {
            Self::Owned(owned) => match owned {
                MleOwned::Base(v) => {
                    *self = Mle::Owned(MleOwned::Extension(fold_multilinear(
                        v,
                        weights,
                        &|a, b| b * a,
                    )))
                }
                MleOwned::Extension(v) => fold_multilinear_in_place(v, weights),
                MleOwned::BasePacked(v) => {
                    *self = Mle::Owned(MleOwned::ExtensionPacked(fold_multilinear(
                        v,
                        &weights
                            .iter()
                            .map(|&w| EFPacking::<EF>::from(w))
                            .collect::<Vec<_>>(),
                        &|a, b| b * a,
                    )))
                }
                MleOwned::ExtensionPacked(v) => fold_multilinear_in_place(v, weights),
            },
            Self::Ref(poly) => match poly {
                MleRef::Base(v) => {
                    *self = Mle::Owned(MleOwned::Extension(fold_multilinear(
                        v,
                        weights,
                        &|a, b| b * a,
                    )))
                }
                MleRef::Extension(v) => {
                    *self = Mle::Owned(MleOwned::Extension(fold_multilinear(
                        v,
                        weights,
                        &|a, b| b * a,
                    )))
                }
                MleRef::BasePacked(v) => {
                    *self = Mle::Owned(MleOwned::ExtensionPacked(fold_multilinear(
                        v,
                        &weights
                            .iter()
                            .map(|&w| EFPacking::<EF>::from(w))
                            .collect::<Vec<_>>(),
                        &|a, b| b * a,
                    )))
                }
                MleRef::ExtensionPacked(v) => {
                    *self = Mle::Owned(MleOwned::ExtensionPacked(fold_multilinear(
                        v,
                        weights,
                        &|a, b| a * b,
                    )))
                }
            },
        }
    }
}
