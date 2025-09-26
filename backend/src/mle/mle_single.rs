use crate::*;
use fiat_shamir::*;
use p3_field::ExtensionField;
use p3_field::PackedValue;
use p3_util::log2_strict_usize;

#[derive(Debug, Clone)]
pub enum Mle<EF: ExtensionField<PF<EF>>> {
    Base(Vec<PF<EF>>),
    Extension(Vec<EF>),
    PackedBase(Vec<PFPacking<EF>>),
    ExtensionPacked(Vec<EFPacking<EF>>),
}





impl<EF: ExtensionField<PF<EF>>> Mle<EF> {
    pub const fn packed_len(&self) -> usize {
        match self {
            Self::Base(v) => v.len(),
            Self::Extension(v) => v.len(),
            Self::PackedBase(v) => v.len(),
            Self::ExtensionPacked(v) => v.len(),
        }
    }

    pub const fn unpacked_len(&self) -> usize {
        let mut res = self.packed_len();
        if self.is_packed() {
            res *= packing_width::<EF>();
        }
        res
    }

    pub fn n_vars(&self) -> usize {
        log2_strict_usize(self.unpacked_len())
    }

    pub fn truncate(&mut self, new_packed_len: usize) {
        match self {
            Self::Base(v) => v.truncate(new_packed_len),
            Self::Extension(v) => v.truncate(new_packed_len),
            Self::PackedBase(v) => v.truncate(new_packed_len),
            Self::ExtensionPacked(v) => v.truncate(new_packed_len),
        }
    }

    pub const fn is_packed(&self) -> bool {
        match self {
            Self::Base(_) | Self::Extension(_) => false,
            Self::PackedBase(_) | Self::ExtensionPacked(_) => true,
        }
    }

    pub const fn as_base(&self) -> Option<&Vec<PF<EF>>> {
        match self {
            Self::Base(b) => Some(b),
            _ => None,
        }
    }

    pub const fn as_extension(&self) -> Option<&Vec<EF>> {
        match self {
            Self::Extension(e) => Some(e),
            _ => None,
        }
    }

    pub const fn as_packed_base(&self) -> Option<&Vec<PFPacking<EF>>> {
        match self {
            Self::PackedBase(pb) => Some(pb),
            _ => None,
        }
    }

    pub const fn as_extension_packed(&self) -> Option<&Vec<EFPacking<EF>>> {
        match self {
            Self::ExtensionPacked(ep) => Some(ep),
            _ => None,
        }
    }

    pub fn evaluate(&self, point: &MultilinearPoint<EF>) -> EF {
        match self {
            Self::Base(pol) => pol.evaluate(point),
            Self::Extension(pol) => pol.evaluate(point),
            Self::PackedBase(pol) => PFPacking::<EF>::unpack_slice(pol).evaluate(point),
            Self::ExtensionPacked(pol) => eval_packed(pol, &point.0),
        }
    }
}