use std::any::TypeId;

use backend::PFPacking;
use p3_commit::BatchOpeningRef;
use p3_commit::ExtensionMmcs;
use p3_commit::Mmcs;
use p3_field::ExtensionField;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{
    KoalaBear, Poseidon2KoalaBear, QuinticExtensionFieldKB, SexticExtensionFieldKB,
    default_koalabear_poseidon2_16,
};
use p3_matrix::Dimensions;
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_merkle_tree::MerkleTree;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

// Ugly file, but this avoids a lot of generic parameters in other places:

pub const DIGEST_ELEMS: usize = 8;

pub(crate) type RoundMerkleTree<F, EF> =
    MerkleTree<F, F, FlatMatrixView<F, EF, DenseMatrix<EF>>, DIGEST_ELEMS>;

type MerkleHashKoalaBear = PaddingFreeSponge<Poseidon2KoalaBear<16>, 16, 8, 8>; // leaf hashing
type MerkleCompressKoalaBear = TruncatedPermutation<Poseidon2KoalaBear<16>, 2, 8, 16>; // 2-to-1 compression
type MerkleTreeMmcsKoalaBear = MerkleTreeMmcs<
    PFPacking<KoalaBear>,
    PFPacking<KoalaBear>,
    MerkleHashKoalaBear,
    MerkleCompressKoalaBear,
    DIGEST_ELEMS,
>;
type KoalaBearExtensionMmcs<EF> = ExtensionMmcs<KoalaBear, EF, MerkleTreeMmcsKoalaBear>;

fn get_koala_bear_mmcs() -> MerkleTreeMmcsKoalaBear {
    let merkle_leaf_hash = MerkleHashKoalaBear::new(default_koalabear_poseidon2_16());
    let merkle_compress = MerkleCompressKoalaBear::new(default_koalabear_poseidon2_16());
    MerkleTreeMmcsKoalaBear::new(merkle_leaf_hash, merkle_compress)
}

fn get_koala_bear_extension_mmcs<EF: ExtensionField<KoalaBear>>() -> KoalaBearExtensionMmcs<EF> {
    KoalaBearExtensionMmcs::new(get_koala_bear_mmcs())
}

pub(crate) fn merkle_commit<F: Field, EF: ExtensionField<F>>(
    matrix: DenseMatrix<EF>,
) -> ([F; DIGEST_ELEMS], RoundMerkleTree<F, EF>) {
    if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, QuinticExtensionFieldKB)>() {
        let matrix =
            unsafe { std::mem::transmute::<_, DenseMatrix<QuinticExtensionFieldKB>>(matrix) };
        let (root, merkle_tree) =
            get_koala_bear_extension_mmcs::<QuinticExtensionFieldKB>().commit_matrix(matrix);
        let root = unsafe { std::mem::transmute_copy::<_, [F; DIGEST_ELEMS]>(&root) };
        let merkle_tree = unsafe { std::mem::transmute::<_, RoundMerkleTree<F, EF>>(merkle_tree) };
        (root, merkle_tree)
    } else if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, KoalaBear)>() {
        let matrix = unsafe { std::mem::transmute::<_, DenseMatrix<KoalaBear>>(matrix) };
        let (root, merkle_tree) = get_koala_bear_mmcs().commit_matrix(matrix);
        let root = unsafe { std::mem::transmute_copy::<_, [F; DIGEST_ELEMS]>(&root) };
        let merkle_tree = unsafe { std::mem::transmute::<_, RoundMerkleTree<F, EF>>(merkle_tree) };
        (root, merkle_tree)
    } else if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, SexticExtensionFieldKB)>() {
        let matrix =
            unsafe { std::mem::transmute::<_, DenseMatrix<SexticExtensionFieldKB>>(matrix) };
        let (root, merkle_tree) =
            get_koala_bear_extension_mmcs::<SexticExtensionFieldKB>().commit_matrix(matrix);
        let root = unsafe { std::mem::transmute_copy::<_, [F; DIGEST_ELEMS]>(&root) };
        let merkle_tree = unsafe { std::mem::transmute::<_, RoundMerkleTree<F, EF>>(merkle_tree) };
        (root, merkle_tree)
    } else if TypeId::of::<(F, EF)>()
        == TypeId::of::<(KoalaBear, BinomialExtensionField<KoalaBear, 4>)>()
    {
        let matrix = unsafe {
            std::mem::transmute::<_, DenseMatrix<BinomialExtensionField<KoalaBear, 4>>>(matrix)
        };
        let (root, merkle_tree) =
            get_koala_bear_extension_mmcs::<BinomialExtensionField<KoalaBear, 4>>()
                .commit_matrix(matrix);
        let root = unsafe { std::mem::transmute_copy::<_, [F; DIGEST_ELEMS]>(&root) };
        let merkle_tree = unsafe { std::mem::transmute::<_, RoundMerkleTree<F, EF>>(merkle_tree) };
        (root, merkle_tree)
    } else {
        unimplemented!()
    }
}

pub(crate) fn merkle_open<F: Field, EF: ExtensionField<F>>(
    merkle_tree: &RoundMerkleTree<F, EF>,
    index: usize,
) -> (Vec<EF>, Vec<[F; DIGEST_ELEMS]>) {
    if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, QuinticExtensionFieldKB)>() {
        let merkle_tree = unsafe {
            std::mem::transmute::<_, &RoundMerkleTree<KoalaBear, QuinticExtensionFieldKB>>(
                merkle_tree,
            )
        };
        let mut batch_opening = get_koala_bear_extension_mmcs::<QuinticExtensionFieldKB>()
            .open_batch(index, merkle_tree);
        let leaf = std::mem::take(&mut batch_opening.opened_values[0]);
        let proof = batch_opening.opening_proof;
        let leaf = unsafe { std::mem::transmute::<_, Vec<EF>>(leaf) };
        let proof = unsafe { std::mem::transmute::<_, Vec<[F; DIGEST_ELEMS]>>(proof) };
        (leaf, proof)
    } else if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, KoalaBear)>() {
        let merkle_tree = unsafe {
            std::mem::transmute::<_, &RoundMerkleTree<KoalaBear, KoalaBear>>(merkle_tree)
        };
        let mut batch_opening =
            get_koala_bear_extension_mmcs::<KoalaBear>().open_batch(index, merkle_tree);
        let leaf = std::mem::take(&mut batch_opening.opened_values[0]);
        let proof = batch_opening.opening_proof;
        let leaf = unsafe { std::mem::transmute::<_, Vec<EF>>(leaf) };
        let proof = unsafe { std::mem::transmute::<_, Vec<[F; DIGEST_ELEMS]>>(proof) };
        (leaf, proof)
    } else if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, SexticExtensionFieldKB)>() {
        let merkle_tree = unsafe {
            std::mem::transmute::<_, &RoundMerkleTree<KoalaBear, SexticExtensionFieldKB>>(
                merkle_tree,
            )
        };
        let mut batch_opening = get_koala_bear_extension_mmcs::<SexticExtensionFieldKB>()
            .open_batch(index, merkle_tree);
        let leaf = std::mem::take(&mut batch_opening.opened_values[0]);
        let proof = batch_opening.opening_proof;
        let leaf = unsafe { std::mem::transmute::<_, Vec<EF>>(leaf) };
        let proof = unsafe { std::mem::transmute::<_, Vec<[F; DIGEST_ELEMS]>>(proof) };
        (leaf, proof)
    } else if TypeId::of::<(F, EF)>()
        == TypeId::of::<(KoalaBear, BinomialExtensionField<KoalaBear, 4>)>()
    {
        let merkle_tree = unsafe {
            std::mem::transmute::<
                _,
                &RoundMerkleTree<KoalaBear, BinomialExtensionField<KoalaBear, 4>>,
            >(merkle_tree)
        };
        let mut batch_opening =
            get_koala_bear_extension_mmcs::<BinomialExtensionField<KoalaBear, 4>>()
                .open_batch(index, merkle_tree);
        let leaf = std::mem::take(&mut batch_opening.opened_values[0]);
        let proof = batch_opening.opening_proof;
        let leaf = unsafe { std::mem::transmute::<_, Vec<EF>>(leaf) };
        let proof = unsafe { std::mem::transmute::<_, Vec<[F; DIGEST_ELEMS]>>(proof) };
        (leaf, proof)
    } else {
        unimplemented!()
    }
}

pub(crate) fn merkle_verify<F: Field, EF: ExtensionField<F>>(
    merkle_root: [F; DIGEST_ELEMS],
    index: usize,
    dimension: Dimensions,
    data: Vec<EF>,
    proof: &Vec<[F; DIGEST_ELEMS]>,
) -> bool {
    if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, QuinticExtensionFieldKB)>() {
        let merkle_root =
            unsafe { std::mem::transmute_copy::<_, [KoalaBear; DIGEST_ELEMS]>(&merkle_root) };
        let data = unsafe { std::mem::transmute::<_, Vec<QuinticExtensionFieldKB>>(data) };
        let proof = unsafe { std::mem::transmute::<_, &Vec<[KoalaBear; DIGEST_ELEMS]>>(proof) };
        get_koala_bear_extension_mmcs::<QuinticExtensionFieldKB>()
            .verify_batch(
                &merkle_root.into(),
                &[dimension],
                index,
                BatchOpeningRef {
                    opened_values: &[data],
                    opening_proof: proof,
                },
            )
            .is_ok()
    } else if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, KoalaBear)>() {
        let merkle_root =
            unsafe { std::mem::transmute_copy::<_, [KoalaBear; DIGEST_ELEMS]>(&merkle_root) };
        let data = unsafe { std::mem::transmute::<_, Vec<KoalaBear>>(data) };
        let proof = unsafe { std::mem::transmute::<_, &Vec<[KoalaBear; DIGEST_ELEMS]>>(proof) };
        get_koala_bear_extension_mmcs::<KoalaBear>()
            .verify_batch(
                &merkle_root.into(),
                &[dimension],
                index,
                BatchOpeningRef {
                    opened_values: &[data],
                    opening_proof: proof,
                },
            )
            .is_ok()
    } else if TypeId::of::<(F, EF)>() == TypeId::of::<(KoalaBear, SexticExtensionFieldKB)>() {
        let merkle_root =
            unsafe { std::mem::transmute_copy::<_, [KoalaBear; DIGEST_ELEMS]>(&merkle_root) };
        let data = unsafe { std::mem::transmute::<_, Vec<SexticExtensionFieldKB>>(data) };
        let proof = unsafe { std::mem::transmute::<_, &Vec<[KoalaBear; DIGEST_ELEMS]>>(proof) };
        get_koala_bear_extension_mmcs::<SexticExtensionFieldKB>()
            .verify_batch(
                &merkle_root.into(),
                &[dimension],
                index,
                BatchOpeningRef {
                    opened_values: &[data],
                    opening_proof: proof,
                },
            )
            .is_ok()
    } else if TypeId::of::<(F, EF)>()
        == TypeId::of::<(KoalaBear, BinomialExtensionField<KoalaBear, 4>)>()
    {
        let merkle_root =
            unsafe { std::mem::transmute_copy::<_, [KoalaBear; DIGEST_ELEMS]>(&merkle_root) };
        let data =
            unsafe { std::mem::transmute::<_, Vec<BinomialExtensionField<KoalaBear, 4>>>(data) };
        let proof = unsafe { std::mem::transmute::<_, &Vec<[KoalaBear; DIGEST_ELEMS]>>(proof) };
        get_koala_bear_extension_mmcs::<BinomialExtensionField<KoalaBear, 4>>()
            .verify_batch(
                &merkle_root.into(),
                &[dimension],
                index,
                BatchOpeningRef {
                    opened_values: &[data],
                    opening_proof: proof,
                },
            )
            .is_ok()
    } else {
        unimplemented!()
    }
}
