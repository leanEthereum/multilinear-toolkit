use fiat_shamir::*;
use p3_field::ExtensionField;

use crate::{MleRef, MultilinearPoint};

#[derive(Debug, Clone)]
pub enum MleOwned<EF: ExtensionField<PF<EF>>> {
    Base(Vec<PF<EF>>),
    Extension(Vec<EF>),
    PackedBase(Vec<PFPacking<EF>>),
    ExtensionPacked(Vec<EFPacking<EF>>),
}

impl<EF: ExtensionField<PF<EF>>> Default for MleOwned<EF> {
    fn default() -> Self {
        Self::Base(vec![])
    }
}

impl<EF: ExtensionField<PF<EF>>> MleOwned<EF> {
    pub fn by_ref<'a>(&'a self) -> MleRef<'a, EF> {
        match self {
            Self::Base(v) => MleRef::Base(v),
            Self::Extension(v) => MleRef::Extension(v),
            Self::PackedBase(v) => MleRef::BasePacked(v),
            Self::ExtensionPacked(v) => MleRef::ExtensionPacked(v),
        }
    }

    pub fn truncate(&mut self, n: usize) {
        match self {
            Self::Base(v) => v.truncate(n),
            Self::Extension(v) => v.truncate(n),
            Self::PackedBase(v) => v.truncate(n),
            Self::ExtensionPacked(v) => v.truncate(n),
        }
    }

    pub fn as_base(&self) -> Option<&[PF<EF>]> {
        match self {
            Self::Base(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_extension(&self) -> Option<&[EF]> {
        match self {
            Self::Extension(e) => Some(e),
            _ => None,
        }
    }

    pub fn as_packed_base(&self) -> Option<&[PFPacking<EF>]> {
        match self {
            Self::PackedBase(pb) => Some(pb),
            _ => None,
        }
    }

    pub fn as_extension_packed(&self) -> Option<&[EFPacking<EF>]> {
        match self {
            Self::ExtensionPacked(ep) => Some(ep),
            _ => None,
        }
    }

    pub fn as_extension_packed_mut(&mut self) -> Option<&mut Vec<EFPacking<EF>>> {
        match self {
            Self::ExtensionPacked(ep) => Some(ep),
            _ => None,
        }
    }

    pub fn into_base(self) -> Option<Vec<PF<EF>>> {
        match self {
            Self::Base(b) => Some(b),
            _ => None,
        }
    }

    pub fn into_extension(self) -> Option<Vec<EF>> {
        match self {
            Self::Extension(e) => Some(e),
            _ => None,
        }
    }

    pub fn into_base_backed(self) -> Option<Vec<PFPacking<EF>>> {
        match self {
            Self::PackedBase(pb) => Some(pb),
            _ => None,
        }
    }

    pub fn into_extension_packed(self) -> Option<Vec<EFPacking<EF>>> {
        match self {
            Self::ExtensionPacked(ep) => Some(ep),
            _ => None,
        }
    }

    pub fn evaluate(&self, point: &MultilinearPoint<EF>) -> EF {
        self.by_ref().evaluate(point)
    }
}
