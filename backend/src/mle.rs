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

#[derive(Debug)]
pub enum MleGroupOwned<EF: ExtensionField<PF<EF>>> {
    Base(Vec<Vec<PF<EF>>>),
    Extension(Vec<Vec<EF>>),
    BasePacked(Vec<Vec<PFPacking<EF>>>),
    ExtensionPacked(Vec<Vec<EFPacking<EF>>>),
}

#[derive(Clone, Debug)]
pub enum MleGroupRef<'a, EF: ExtensionField<PF<EF>>> {
    Base(Vec<&'a [PF<EF>]>),
    Extension(Vec<&'a [EF]>),
    BasePacked(Vec<&'a [PFPacking<EF>]>),
    ExtensionPacked(Vec<&'a [EFPacking<EF>]>),
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
}

impl<'a, EF: ExtensionField<PF<EF>>> MleGroupRef<'a, EF> {
    pub const fn group_size(&self) -> usize {
        match self {
            Self::Base(v) => v.len(),
            Self::Extension(v) => v.len(),
            Self::BasePacked(v) => v.len(),
            Self::ExtensionPacked(v) => v.len(),
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

    pub const fn is_packed(&self) -> bool {
        match self {
            Self::BasePacked(_) | Self::ExtensionPacked(_) => true,
            Self::Base(_) | Self::Extension(_) => false,
        }
    }

    pub const fn as_base(&self) -> Option<&Vec<&'a [PF<EF>]>> {
        match self {
            Self::Base(b) => Some(b),
            _ => None,
        }
    }

    pub const fn as_extension(&self) -> Option<&Vec<&'a [EF]>> {
        match self {
            Self::Extension(e) => Some(e),
            _ => None,
        }
    }

    pub const fn as_packed_base(&self) -> Option<&Vec<&'a [PFPacking<EF>]>> {
        match self {
            Self::BasePacked(pb) => Some(pb),
            _ => None,
        }
    }

    pub const fn as_extension_packed(&self) -> Option<&Vec<&'a [EFPacking<EF>]>> {
        match self {
            Self::ExtensionPacked(ep) => Some(ep),
            _ => None,
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

    pub fn pack(&self) -> MleGroup<'a, EF> {
        match self {
            Self::Base(base) => MleGroupRef::BasePacked(
                base.iter()
                    .map(|v| PFPacking::<EF>::pack_slice(v))
                    .collect(),
            )
            .into(),
            Self::Extension(ext) => {
                // the only case where there is real work
                MleGroupOwned::ExtensionPacked(ext.iter().map(|v| pack_extension(v)).collect())
                    .into()
            }
            Self::BasePacked(_) | Self::ExtensionPacked(_) => self.clone().into(),
        }
    }

    // Clone everything in the group, should not be used when n_vars is large
    pub fn unpack(&self) -> MleGroupOwned<EF> {
        match self {
            Self::Base(pols) => MleGroupOwned::Base(pols.iter().map(|v| v.to_vec()).collect()),
            Self::Extension(pols) => {
                MleGroupOwned::Extension(pols.iter().map(|v| v.to_vec()).collect())
            }
            Self::BasePacked(pols) => MleGroupOwned::Base(
                pols.iter()
                    .map(|v| PFPacking::<EF>::unpack_slice(v).to_vec())
                    .collect(),
            ),
            Self::ExtensionPacked(pols) => {
                MleGroupOwned::Extension(pols.iter().map(|v| unpack_extension(v)).collect())
            }
        }
    }

    pub fn fold(&self, scalars: &[EF]) -> MleGroupOwned<EF> {
        match self {
            Self::Base(pols) => {
                MleGroupOwned::Extension(batch_fold_multilinears(pols, scalars, |a, b| b * a))
            }
            Self::Extension(pols) => {
                MleGroupOwned::Extension(batch_fold_multilinears(pols, scalars, |a, b| b * a))
            }
            Self::BasePacked(pols) => {
                let scalars_packed = scalars
                    .iter()
                    .map(|&s| EFPacking::<EF>::from(s))
                    .collect::<Vec<_>>();
                MleGroupOwned::ExtensionPacked(batch_fold_multilinears(
                    pols,
                    &scalars_packed,
                    |a, b| b * a,
                ))
            }
            Self::ExtensionPacked(pols) => {
                MleGroupOwned::ExtensionPacked(batch_fold_multilinears(pols, scalars, |a, b| a * b))
            }
        }
    }
}
