import io
import pickle
import string
import os

import numpy as np
import torch
from torch.utils import data
import collections

from foldflow.data import residue_constants
from openfold.utils import rigid_utils
from typing import Dict, List, Tuple, Union, Any
from Bio import PDB
from Bio.PDB.Chain import Chain
import dataclasses
from foldflow.data.protein import Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + " "
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = {i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)}

CHAIN_FEATS = ["atom_positions", "aatype", "atom_mask", "residue_index", "b_factors"]
UNPADDED_FEATS = [
    "t",
    "rot_vectorfield_scaling",
    "trans_vectorfield_scaling",
    "t_seq",
    "t_struct",
]
RIGID_FEATS = ["rigids_0", "rigids_t"]
PAIR_FEATS = ["rel_rots"]


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, "rb") as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(
                    f"Failed to read {read_path}. First error: {e}\n Second error: {e2}"
                )
            raise (e)


def parse_chain_feats(chain_feats, scale_factor=1.0):
    ca_idx = residue_constants.atom_order["CA"]
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, ca_idx]
    bb_pos = chain_feats["atom_positions"][:, ca_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
    centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, ca_idx]
    return chain_feats


def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)


def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    pad_rigid = rigid_utils.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False
    )
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)


def pad_feats(raw_feats, max_len, use_torch=False):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS
    }
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    return padded_feats


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def length_batching_multi_gpu(
    np_dicts: List[Dict[str, np.ndarray]],
    max_squared_res: int,
    num_gpus: int,
):
    def get_len(x):
        return x["res_mask"].shape[0]

    # get_len = lambda x: x['res_mask'].shape[0]
    # Filter out Nones! (Hacky solution to not sample more examples than necessary)
    # Split per GPU based on num_gpus

    np_dicts = [x for x in np_dicts if x is not None]

    dicts_by_length = [(get_len(x), x) for x in np_dicts]

    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    max_batch_examples = max(int(max_squared_res // max_len**2), 1)
    pad_example = lambda x: pad_feats(x, max_len)
    padded_batch = [pad_example(x) for (_, x) in length_sorted[:max_batch_examples]]
    return torch.utils.data.default_collate(padded_batch)


def concat_np_features(np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def possible_tuple_length_batching_multi_gpu(
    x: Union[List[Dict[str, np.ndarray]], Tuple[List[Dict[str, np.ndarray]], str]],
    max_squared_res: int,
    num_gpus: int,
):
    if type(x[0]) == tuple:
        # Assume this is a validation dataset of the second type
        return length_batching_multi_gpu(
            [y[0] for y in x], max_squared_res, num_gpus
        ), [y[1] for y in x]
    else:
        return length_batching_multi_gpu(x, max_squared_res, num_gpus)


def possible_tuple_length_batching(
    x: Union[List[Dict[str, np.ndarray]], Tuple[List[Dict[str, np.ndarray]], str]],
    max_squared_res: int,
):
    if type(x[0]) == tuple:
        # Assume this is a validation dataset of the second type
        return length_batching([y[0] for y in x], max_squared_res), [y[1] for y in x]
    else:
        return length_batching(x, max_squared_res)


def length_batching(
    np_dicts: List[Dict[str, np.ndarray]],
    max_squared_res: int,
):
    def get_len(x):
        return x["res_mask"].shape[0]

    # get_len = lambda x: x['res_mask'].shape[0]
    # Filter out Nones! (Hacky solution to not sample more examples than necessary)

    np_dicts = [x for x in np_dicts if x is not None]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]

    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    if len(length_sorted) == 0:
        return torch.utils.data.default_collate([{"dummy_batch": np.random.rand(100)}])

    max_len = length_sorted[0][0]
    max_batch_examples = max(int(max_squared_res // max_len**2), 1)
    pad_example = lambda x: pad_feats(x, max_len)

    keep = length_sorted[:max_batch_examples]
    padded_batch = [pad_example(x) for (_, x) in keep]

    return torch.utils.data.default_collate(padded_batch)


def create_data_loader(
    torch_dataset: data.Dataset,
    batch_size,
    shuffle,
    sampler=None,
    num_workers=0,
    np_collate=False,
    max_squared_res=1e6,
    length_batch=False,
    drop_last=False,
    prefetch_factor=2,
    num_gpus=1,
):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)
    elif length_batch:
        if num_gpus > 1:
            collate_fn = lambda x: possible_tuple_length_batching_multi_gpu(
                x, max_squared_res=max_squared_res, num_gpus=num_gpus
            )
        else:
            collate_fn = lambda x: possible_tuple_length_batching(
                x, max_squared_res=max_squared_res
            )
    else:
        collate_fn = None

    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=True,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context="fork"
        if num_workers != 0
        else None,  # TODO Try without. Doesn't seem to matter
    )


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[
        ..., None
    ]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def move_to_np(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    if isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(x)}.")


def aatype_to_seq(aatype):
    return "".join([residue_constants.restypes_with_x[x] for x in aatype])


def write_pkl(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_checkpoint(
    ckpt_path: str,
    model,
    conf,
    optimizer,
    epoch,
    step,
    logger=None,
    use_torch=True,
):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    # for fname in os.listdir(ckpt_dir):
    # if '.pkl' in fname or '.pth' in fname:
    #    os.remove(os.path.join(ckpt_dir, fname))
    if logger is not None:
        logger.info(f"Serializing experiment state to {ckpt_path}")
    else:
        print(f"Serializing experiment state to {ckpt_path}")
    write_pkl(
        ckpt_path,
        {
            "model": model,
            "conf": conf,
            "optimizer": optimizer,
            "epoch": epoch,
            "step": step,
        },
        use_torch=use_torch,
    )


def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.0
            res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors),
    )


def parse_pdb_feats(
    pdb_name: str,
    pdb_path: str,
    scale_factor=1.0,
    # TODO: Make the default behaviour read all chains.
    chain_id="A",
):
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {chain.id: chain for chain in structure.get_chains()}

    def _process_chain_id(x):
        chain_prot = process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Process features
        feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        return parse_chain_feats(feat_dict, scale_factor=scale_factor)

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {x: _process_chain_id(x) for x in chain_id}
    elif chain_id is None:
        return {x: _process_chain_id(x) for x in struct_chains}
    else:
        raise ValueError(f"Unrecognized chain list {chain_id}")


def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected
