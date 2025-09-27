use fiat_shamir::*;
use p3_field::ExtensionField;

use crate::{MleRef, MultilinearPoint};

#[derive(Debug, Clone)]
pub enum MleOwned<EF: ExtensionField<PF<EF>>> {
    Base(Vec<PF<EF>>),
    Extension(Vec<EF>),
    BasePacked(Vec<PFPacking<EF>>),
    ExtensionPacked(Vec<EFPacking<EF>>),
}

impl<EF: ExtensionField<PF<EF>>> MleOwned<EF> {
    pub fn by_ref<'a>(&'a self) -> MleRef<'a, EF> {
        match self {
            Self::Base(v) => MleRef::Base(v),
            Self::Extension(v) => MleRef::Extension(v),
            Self::BasePacked(v) => MleRef::BasePacked(v),
            Self::ExtensionPacked(v) => MleRef::ExtensionPacked(v),
        }
    }

    pub fn truncate(&mut self, n: usize) {
        match self {
            Self::Base(v) => v.truncate(n),
            Self::Extension(v) => v.truncate(n),
            Self::BasePacked(v) => v.truncate(n),
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
            Self::BasePacked(pb) => Some(pb),
            _ => None,
        }
    }

    pub fn as_extension_packed(&self) -> Option<&[EFPacking<EF>]> {
        match self {
            Self::ExtensionPacked(ep) => Some(ep),
            _ => None,
        }
    }

    pub fn evaluate(&self, point: &MultilinearPoint<EF>) -> EF {
        self.by_ref().evaluate(point)
    }
}
