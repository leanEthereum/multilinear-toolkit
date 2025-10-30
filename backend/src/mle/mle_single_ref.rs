use crate::*;
use fiat_shamir::*;
use p3_field::ExtensionField;
use p3_field::PackedValue;
use p3_util::log2_strict_usize;

#[derive(Debug, Clone)]
pub enum MleRef<'a, EF: ExtensionField<PF<EF>>> {
    Base(&'a [PF<EF>]),
    Extension(&'a [EF]),
    BasePacked(&'a [PFPacking<EF>]),
    ExtensionPacked(&'a [EFPacking<EF>]),
}

impl<'a, EF: ExtensionField<PF<EF>>> MleRef<'a, EF> {
    pub const fn packed_len(&self) -> usize {
        match self {
            Self::Base(v) => v.len(),
            Self::Extension(v) => v.len(),
            Self::BasePacked(v) => v.len(),
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

    pub const fn is_packed(&self) -> bool {
        match self {
            Self::Base(_) | Self::Extension(_) => false,
            Self::BasePacked(_) | Self::ExtensionPacked(_) => true,
        }
    }

    pub const fn as_base(&self) -> Option<&[PF<EF>]> {
        match self {
            Self::Base(b) => Some(b),
            _ => None,
        }
    }

    pub const fn as_extension(&self) -> Option<&[EF]> {
        match self {
            Self::Extension(e) => Some(e),
            _ => None,
        }
    }

    pub const fn as_packed_base(&self) -> Option<&[PFPacking<EF>]> {
        match self {
            Self::BasePacked(pb) => Some(pb),
            _ => None,
        }
    }

    pub const fn as_extension_packed(&self) -> Option<&[EFPacking<EF>]> {
        match self {
            Self::ExtensionPacked(ep) => Some(ep),
            _ => None,
        }
    }

    pub fn evaluate(&self, point: &MultilinearPoint<EF>) -> EF {
        match self {
            Self::Base(pol) => pol.evaluate(point),
            Self::Extension(pol) => pol.evaluate(point),
            Self::BasePacked(pol) => PFPacking::<EF>::unpack_slice(pol).evaluate(point),
            Self::ExtensionPacked(pol) => eval_packed::<_, true>(pol, &point.0),
        }
    }

    pub fn pack(&self) -> Mle<'a, EF> {
        match self {
            Self::Base(v) => Mle::Ref(MleRef::BasePacked(PFPacking::<EF>::pack_slice(v))),
            Self::Extension(v) => Mle::Owned(MleOwned::ExtensionPacked(pack_extension(v))),
            Self::BasePacked(_) => Mle::Ref(self.clone()),
            Self::ExtensionPacked(_) => Mle::Ref(self.clone()),
        }
    }

    pub fn unpack(&self) -> Mle<'a, EF> {
        match self {
            Self::Base(v) => Mle::Ref(MleRef::Base(v)),
            Self::Extension(v) => Mle::Ref(MleRef::Extension(v)),
            Self::BasePacked(pb) => Mle::Ref(MleRef::Base(PFPacking::<EF>::unpack_slice(pb))),
            Self::ExtensionPacked(ep) => Mle::Owned(MleOwned::Extension(unpack_extension(ep))),
        }
    }

    pub fn pack_if(&self, cond: bool) -> Mle<'a, EF> {
        if cond {
            self.pack()
        } else {
            Mle::Ref(self.clone())
        }
    }

    pub fn clone_to_owned(&self) -> MleOwned<EF> {
        match self {
            Self::Base(v) => MleOwned::Base(v.to_vec()),
            Self::Extension(v) => MleOwned::Extension(v.to_vec()),
            Self::BasePacked(pb) => MleOwned::BasePacked(pb.to_vec()),
            Self::ExtensionPacked(ep) => MleOwned::ExtensionPacked(ep.to_vec()),
        }
    }

    pub fn fold(&self, scalars: &[EF]) -> MleOwned<EF> {
        match self {
            Self::Base(pols) => MleOwned::Extension(fold_multilinear(pols, scalars, &|a, b| b * a)),
            Self::Extension(pols) => {
                MleOwned::Extension(fold_multilinear(pols, scalars, &|a, b| b * a))
            }
            Self::BasePacked(pols) => {
                let scalars_packed = scalars
                    .iter()
                    .map(|&s| EFPacking::<EF>::from(s))
                    .collect::<Vec<_>>();
                MleOwned::ExtensionPacked(fold_multilinear(pols, &scalars_packed, &|a, b| b * a))
            }
            Self::ExtensionPacked(pols) => {
                MleOwned::ExtensionPacked(fold_multilinear(pols, scalars, &|a, b| a * b))
            }
        }
    }
}
