use crate::*;
use fiat_shamir::*;
use p3_field::ExtensionField;
use p3_util::log2_strict_usize;

#[derive(Debug)]
pub enum MleGroupOwned<EF: ExtensionField<PF<EF>>> {
    Base(Vec<Vec<PF<EF>>>),
    Extension(Vec<Vec<EF>>),
    BasePacked(Vec<Vec<PFPacking<EF>>>),
    ExtensionPacked(Vec<Vec<EFPacking<EF>>>),
}

impl<EF: ExtensionField<PF<EF>>> MleGroupOwned<EF> {
    pub fn as_extension_mut(&mut self) -> Option<&mut Vec<Vec<EF>>> {
        match self {
            Self::Extension(e) => Some(e),
            _ => None,
        }
    }

    pub fn by_ref<'a>(&'a self) -> MleGroupRef<'a, EF> {
        match self {
            Self::Base(base) => MleGroupRef::Base(base.iter().map(|v| v.as_slice()).collect()),
            Self::Extension(ext) => {
                MleGroupRef::Extension(ext.iter().map(|v| v.as_slice()).collect())
            }
            Self::BasePacked(packed_base) => {
                MleGroupRef::BasePacked(packed_base.iter().map(|v| v.as_slice()).collect())
            }
            Self::ExtensionPacked(ext_packed) => {
                MleGroupRef::ExtensionPacked(ext_packed.iter().map(|v| v.as_slice()).collect())
            }
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            Self::Base(v) => log2_strict_usize(v[0].len()),
            Self::Extension(v) => log2_strict_usize(v[0].len()),
            Self::BasePacked(v) => log2_strict_usize(v[0].len() * packing_width::<EF>()),
            Self::ExtensionPacked(v) => log2_strict_usize(v[0].len() * packing_width::<EF>()),
        }
    }

    pub const fn n_columns(&self) -> usize {
        match self {
            Self::Base(v) => v.len(),
            Self::Extension(v) => v.len(),
            Self::BasePacked(v) => v.len(),
            Self::ExtensionPacked(v) => v.len(),
        }
    }

    pub fn fold_in_large_field_in_place(&mut self, scalars: &[EF]) {
        match self {
            Self::Base(_) | Self::BasePacked(_) => {
                *self = self.by_ref().fold(scalars);
            }
            Self::Extension(pols) => {
                batch_fold_multilinear_in_place(
                    &mut pols.iter_mut().map(|p| p.as_mut()).collect::<Vec<_>>(),
                    scalars,
                );
            }
            Self::ExtensionPacked(pols) => {
                batch_fold_multilinear_in_place(
                    &mut pols.iter_mut().map(|p| p.as_mut()).collect::<Vec<_>>(),
                    scalars,
                );
            }
        }
    }
}
