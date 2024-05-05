import functools as fn
import logging
import math
import os
import pickle
import random
import time
from functools import partial
from multiprocessing import get_context
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Optional

import lmdb
import numpy as np
import ot as pot
import pandas as pd
import torch
import torch.distributed as dist
import tree
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils import data
from tqdm import tqdm

from foldflow.data import utils as du
from foldflow.utils.rigid_helpers import assemble_rigid_mat, extract_trans_rots_mat
from foldflow.utils.so3_helpers import so3_relative_angle
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils

_BYTES_PER_MEGABYTE = int(1e6)


def get_list_chunk_slices(lst, chunk_size):
    return [(i, i + chunk_size) for i in range(0, len(lst), chunk_size)]


def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y


def get_csv_rows_many(csv, shared_list, idx_slice):
    start_idx, end_idx = tuple(map(lambda x: min(x, len(csv)), idx_slice))
    for idx in tqdm(list(range(start_idx, end_idx))):
        shared_list[idx] = pickle.dumps(get_csv_row(csv, idx))

    print("Finished saving data to pickle")


# @fn.lru_cache(maxsize=100)
def get_csv_row(csv, idx):
    """Get on row of the csv file, and prepare the pdb feature dict.

    Args:
        idx (int): idx of the row
        csv (pd.DataFrame): csv pd.DataFrame

    Returns:
        tuple: dict of the features, ground truth backbone rigid, pdb_name
    """
    # Sample data example.
    example_idx = idx
    csv_row = csv.iloc[example_idx]
    if "pdb_name" in csv_row:
        pdb_name = csv_row["pdb_name"]
    elif "chain_name" in csv_row:
        pdb_name = csv_row["chain_name"]
    else:
        raise ValueError("Need chain identifier.")
    processed_file_path = csv_row["processed_path"]
    chain_feats = _process_csv_row(csv, processed_file_path)

    gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[:, 0]
    flowed_mask = np.ones_like(chain_feats["res_mask"])
    if np.sum(flowed_mask) < 1:
        raise ValueError("Must be flowed")
    fixed_mask = 1 - flowed_mask
    chain_feats["fixed_mask"] = fixed_mask
    chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
    chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

    return chain_feats, gt_bb_rigid, pdb_name, csv_row


def _process_csv_row(csv, processed_file_path):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats["modeled_idx"]
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats["modeled_idx"]
    processed_feats = tree.map_structure(
        lambda x: x[min_idx : (max_idx + 1)], processed_feats
    )

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats["chain_index"]
    res_idx = processed_feats["residue_index"]
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = (
        np.array(random.sample(all_chain_idx, len(all_chain_idx)))
        - np.min(all_chain_idx)
        + 1
    )
    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

    # To speed up processing, only take necessary features
    final_feats = {
        "aatype": chain_feats["aatype"],
        "seq_idx": new_res_idx,
        "chain_idx": new_chain_idx,
        "residx_atom14_to_atom37": chain_feats["residx_atom14_to_atom37"],
        "residue_index": processed_feats["residue_index"],
        "res_mask": processed_feats["bb_mask"],
        "atom37_pos": chain_feats["all_atom_positions"],
        "atom37_mask": chain_feats["all_atom_mask"],
        "atom14_pos": chain_feats["atom14_gt_positions"],
        "rigidgroups_0": chain_feats["rigidgroups_gt_frames"],
        "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
    }

    return final_feats


class PdbDataset(data.Dataset):
    """PDB dataset, with or without OT plan.

    Args:
        data_conf : configuration for the dataset
        gen_model : the model used to generate the data
        is_training : whether the dataset is used for training or validation
        is_OT : whether to use OT pairings
        ot_fn : method to use for OT (exact, sinkhorn). Default is `"exact"`.
        reg : regularization for Sinkhorn OT. Default is `0.05`.
        max_same_res : max number of same length proteins in a batch for OT. Default is `10`.
    """

    def __init__(
        self,
        *,
        data_conf,
        gen_model,
        is_training,
        is_OT=False,  # whether to use OT pairings
        ot_fn="exact",  # method to use for OT
        reg=0.05,  # regularization for OT
        max_same_res=10,  # max number of same length proteins in a batch for OT.
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._init_metadata()

        self._cache_dataset = data_conf.cache_full_dataset
        self._cache_dataset_in_memory = data_conf.cache_dataset_in_memory
        self._cache_path = data_conf.cache_path
        self._store_result_tuples = None
        self._local_cache = None

        if self._cache_dataset:
            # self._build_dataset_cache()
            self._build_dataset_cache_v2()

        # Could be Diffusion, CFM, OT-CFM or SF2M
        self._gen_model = gen_model
        self.is_OT = is_OT
        self.reg = reg
        self._max_same_res = max_same_res
        self._ot_fn = ot_fn.lower()

    @property
    def ot_fn(self):
        # import ot as pot
        if self._ot_fn == "exact":
            return pot.emd
        elif self._ot_fn == "sinkhorn":
            return partial(pot.sinkhorn, reg=self.reg)

    @property
    def max_same_res(self):
        if self.is_OT:
            return self._max_same_res
        else:
            return -1

    @property
    def is_training(self):
        return self._is_training

    @property
    def gen_model(self):
        return self._gen_model

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv
        if (
            filter_conf.allowed_oligomer is not None
            and len(filter_conf.allowed_oligomer) > 0
        ):
            pdb_csv = pdb_csv[
                pdb_csv.oligomeric_detail.isin(filter_conf.allowed_oligomer)
            ]
        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.max_loop_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.coil_percent < filter_conf.max_loop_percent]
        if filter_conf.min_beta_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.strand_percent > filter_conf.min_beta_percent]
        if filter_conf.rog_quantile is not None and filter_conf.rog_quantile > 0.0:
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv, filter_conf.rog_quantile, np.arange(filter_conf.max_len)
            )
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
                lambda x: prot_rog_low_pass[x - 1]
            )
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]
        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[: filter_conf.subset]
        pdb_csv = pdb_csv.sort_values("modeled_seq_len", ascending=False)
        self._create_split(pdb_csv)

    def _build_dataset_cache(self):
        print("Starting to process dataset csv into memory")
        print(f"ROWS {len(self.csv)}")
        # self.csv = self.csv.iloc[:500]
        print(f"Running only {len(self.csv)}")

        st_time = time.time()

        dataset_size = len(self.csv)
        num_chunks = math.ceil(float(dataset_size) / self.data_conf.num_csv_processors)

        idx_chunks = get_list_chunk_slices(list(range(dataset_size)), num_chunks)

        result_tuples = [None] * len(self.csv)
        with SharedMemoryManager() as smm:
            with get_context("spawn").Pool(self.data_conf.num_csv_processors) as pool:
                shared_list = smm.ShareableList(
                    [bytes(3 * _BYTES_PER_MEGABYTE) for _ in range(len(self.csv))]
                )
                partial_fxn = fn.partial(get_csv_rows_many, self.csv, shared_list)
                iterator = enumerate(pool.imap(partial_fxn, idx_chunks))
                for idx, _ in tqdm(iterator, total=len(idx_chunks)):
                    start_idx, end_idx = tuple(
                        map(lambda x: min(x, len(self.csv)), idx_chunks[idx])
                    )

                    for inner_idx in tqdm(range(start_idx, end_idx)):
                        result_tuples[inner_idx] = pickle.loads(shared_list[inner_idx])
                        shared_list[inner_idx] = ""

        def _get_list(idx):
            return list(map(lambda x: x[idx], result_tuples))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chain_ftrs = _get_list(0)
        self.gt_bb_rigid_vals = _get_list(1)
        self.pdb_names = _get_list(2)
        self.csv_rows = _get_list(3)
        print(
            f"Finished processing dataset csv into memory in {time.time() - st_time} seconds"
        )

        print("Finished loading dataset into RAM")

    def _build_dataset_cache_v2(self):
        print(
            f"Starting to process dataset csv into memory "
            f"(cache_dataset_in_memory {self._cache_dataset_in_memory})"
        )
        print(f"ROWS {len(self.csv)}")
        # self.csv = self.csv.iloc[:500]
        print(f"Running only {len(self.csv)}")

        build_local_cache = True
        if os.path.isdir(self._cache_path):
            build_local_cache = False
            print(f"Found local cache @ {self._cache_path}, skipping build")

        # Initialize local cache with lmdb
        self._local_cache = lmdb.open(
            self._cache_path, map_size=(1024**3) * 60
        )  # 1GB * 60

        st_time = time.time()

        if build_local_cache:
            print(f"Building cache and saving @ {self._cache_path}")

            dataset_size = len(self.csv)
            num_chunks = math.ceil(
                float(dataset_size) / self.data_conf.num_csv_processors
            )

            idx_chunks = get_list_chunk_slices(list(range(dataset_size)), num_chunks)

            result_tuples = [None] * len(self.csv)

            pbar = tqdm(total=len(self.csv))
            with self._local_cache.begin(write=True) as txn:
                with SharedMemoryManager() as smm:
                    with get_context("spawn").Pool(
                        self.data_conf.num_csv_processors
                    ) as pool:
                        shared_list = smm.ShareableList(
                            [
                                bytes(3 * _BYTES_PER_MEGABYTE)
                                for _ in range(len(self.csv))
                            ]
                        )
                        partial_fxn = fn.partial(
                            get_csv_rows_many, self.csv, shared_list
                        )
                        iterator = enumerate(pool.imap(partial_fxn, idx_chunks))
                        for idx, _ in iterator:
                            start_idx, end_idx = tuple(
                                map(lambda x: min(x, len(self.csv)), idx_chunks[idx])
                            )
                            # print(f"RUNNING {start_idx} {end_idx} : chunks  {idx_chunks[idx]}")
                            for inner_idx in tqdm(range(start_idx, end_idx)):
                                txn.put(str(inner_idx).encode(), shared_list[inner_idx])

                                if self._cache_dataset_in_memory:
                                    result_tuples[inner_idx] = pickle.loads(
                                        shared_list[inner_idx]
                                    )

                                shared_list[inner_idx] = ""
                                pbar.update(1)
        elif self._cache_dataset_in_memory:
            print(f"Loading cache from local dataset @ {self._cache_path}")
            result_tuples = [None] * len(self.csv)
            with self._local_cache.begin() as txn:
                for ix in range(len(self.csv)):
                    result_tuples[ix] = pickle.loads(txn.get(str(ix).encode()))

        if self._cache_dataset_in_memory:

            def _get_list(idx):
                return list(map(lambda x: x[idx], result_tuples))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.chain_ftrs = _get_list(0)
            self.gt_bb_rigid_vals = _get_list(1)
            self.pdb_names = _get_list(2)
            self.csv_rows = _get_list(3)

        print(
            f"Finished processing dataset csv into memory in {time.time() - st_time} seconds"
        )
        print("Finished loading dataset into RAM")

    def _get_cached_csv_row(self, idx, csv=None):
        if csv is not None:
            # We are going to get the idx row out of the csv -> so we look for true index based on index cl
            idx = csv.iloc[idx]["index"]

        if self._cache_dataset_in_memory:
            return (
                self.chain_ftrs[idx],
                self.gt_bb_rigid_vals[idx],
                self.pdb_names[idx],
                self.csv_rows[idx],
            )
        else:
            return self._get_cached_csv_irow(idx)

    def _get_cached_csv_irow(self, idx, csv=None):

        with self._local_cache.begin() as txn:
            data = txn.get(str(idx).encode())
        return pickle.loads(data)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            all_lengths = np.sort(pdb_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self._data_conf.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len.isin(eval_lengths)]
            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby("modeled_seq_len").sample(
                self._data_conf.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values("modeled_seq_len", ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )

    def _create_flowed_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order["CA"]]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust flow mask sampling method.
        flow_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(flow_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min,
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min, high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            flow_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(f"Unable to generate diffusion mask for {row}")
        return flow_mask

    def __len__(self):
        return len(self.csv)

    def _get_csv_row(self, idx, csv=None):
        """Get on row of the csv file, and prepare the pdb feature dict.

        Args:
            idx (int): idx of the row
            csv (pd.DataFrame): csv pd.DataFrame

        Returns:
            tuple: dict of the features, ground truth backbone rigid, pdb_name
        """
        if self._cache_dataset:
            return self._get_cached_csv_row(idx, csv)
        else:
            if csv is None:
                csv = self.csv

            return get_csv_row(csv, idx)

    def __getitem__(self, idx) -> Any:
        # Custom sampler can return None for idx None.
        # Hacky way to simulate a fixed batch size.
        if idx is None:
            return None

        # print(f"[DEBUG] Train dataset getitem")

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        chain_feats, gt_bb_rigid, pdb_name, csv_row = self._get_csv_row(idx)

        if self.is_training and not self.is_OT:
            # Sample t and flow.
            t = rng.uniform(self._data_conf.min_t, 1.0)
            gen_feats_t = self._gen_model.forward_marginal(
                rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=None
            )
        elif self.is_training and self.is_OT:
            t = rng.uniform(self._data_conf.min_t, 1.0)
            n_res = chain_feats["aatype"].shape[
                0
            ]  # feat['aatype'].shape = (batch, n_res)
            # get a maximum of self.max_same_res proteins with the same length
            subset = self.csv[self.csv["modeled_seq_len"] == n_res]
            n_samples = min(subset.shape[0], self.max_same_res)
            if n_samples == 1 or n_samples == 0:
                # only one sample, we can't do OT
                # self._log.info(f"Only one sample of length {n_res}, skipping OT")
                gen_feats_t = self._gen_model.forward_marginal(
                    rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=None
                )
            else:
                sample_subset = subset.sample(
                    n_samples, replace=True, random_state=0
                ).reset_index(drop=True)

                # get the features, transform them to Rigid, and extract their translation and rotation.
                list_feat = [
                    self._get_csv_row(i, sample_subset)[0] for i in range(n_samples)
                ]
                list_trans_rot = [
                    extract_trans_rots_mat(
                        rigid_utils.Rigid.from_tensor_7(feat["rigids_0"])
                    )
                    for feat in list_feat
                ]
                list_trans, list_rot = zip(*list_trans_rot)

                # stack them and change them to torch.tensor
                sample_trans = torch.stack(
                    [torch.from_numpy(trans) for trans in list_trans]
                )
                sample_rot = torch.stack([torch.from_numpy(rot) for rot in list_rot])

                device = sample_rot.device  # TODO: set the device before that...

                # random matrices on S03.
                rand_rot = torch.tensor(
                    Rotation.random(n_samples * n_res).as_matrix()
                ).to(device=device, dtype=sample_rot.dtype)
                rand_rot = rand_rot.reshape(n_samples, n_res, 3, 3)
                # rand_rot_axis_angle = matrix_to_axis_angle(rand_rot)

                # random translation
                rand_trans = torch.randn(size=(n_samples, n_res, 3)).to(
                    device=device, dtype=sample_trans.dtype
                )

                # compute the ground cost for OT: sum of the cost for S0(3) and R3.
                ground_cost = torch.zeros(n_samples, n_samples).to(device)

                for i in range(n_samples):
                    for j in range(i, n_samples):
                        s03_dist = torch.sum(
                            so3_relative_angle(sample_rot[i], rand_rot[j])
                        )
                        r3_dist = torch.sum(
                            torch.linalg.norm(sample_trans[i] - rand_trans[j], dim=-1)
                        )
                        ground_cost[i, j] = s03_dist**2 + r3_dist**2
                        ground_cost[j, i] = ground_cost[i, j]

                # OT with uniform distributions over the set of pdbs
                a = pot.unif(n_samples, type_as=ground_cost)
                b = pot.unif(n_samples, type_as=ground_cost)
                T = self.ot_fn(
                    a, b, ground_cost
                )  # NOTE: `ground_cost` is the squared distance on SE(3)^N.

                # sample using the plan
                # pick one random indices for the pdb returned by __getitem__
                idx_target = torch.randint(n_samples, (1,))
                pi_target = T[idx_target].squeeze()
                pi_target /= torch.sum(pi_target)
                idx_source = torch.multinomial(pi_target, 1)
                paired_rot = rand_rot[idx_source].squeeze()
                paired_trans = rand_trans[idx_source].squeeze()

                rigids_1 = assemble_rigid_mat(paired_rot, paired_trans)

                gen_feats_t = self._gen_model.forward_marginal(
                    rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=rigids_1
                )

        else:
            t = 1.0
            gen_feats_t = self.gen_model.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                flow_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(gen_feats_t)
        chain_feats["t"] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        final_feats = du.pad_feats(final_feats, csv_row["modeled_seq_len"])
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name


class TrainSampler(data.Sampler):
    def __init__(
        self,
        *,
        data_conf,
        dataset,
        batch_size,
        sample_mode,
        max_squared_res,
        num_gpus,
    ):
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv["index"] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self._max_squared_res = max_squared_res
        self.sampler_len = len(self._dataset_indices) * self._batch_size
        self._num_gpus = num_gpus

        if self._sample_mode in [
            "cluster_length_batch",
            "cluster_time_batch",
            "cluster_time_batch_v2",
        ]:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._log.info(f"Read {self._max_cluster} clusters.")
            self._missing_pdbs = 0

            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]

            self._data_csv["cluster"] = self._data_csv["pdb_name"].map(cluster_lookup)
            num_clusters = len(set(self._data_csv["cluster"]))
            self.sampler_len = num_clusters * self._batch_size
            self._log.info(
                f"Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}"
            )

            # TODO Make sure seq len is modeled_seq_len
            self._data_csv["max_batch_examples"] = self._data_csv[
                "modeled_seq_len"
            ].apply(lambda x: max(int(max_squared_res // x**2), 1))
            self._data_csv_group_clusters = self._data_csv.groupby("cluster")

        # We are assuming we are indexing based on relative position in the csv (with pandas iloc)
        assert np.all(
            self._data_csv["index"].values == np.arange(len(self._data_csv))
        ), "CSV is not sorted by index."

        # breakpoint()

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._data_conf.cluster_path, "r") as f:
            for i, line in enumerate(f):
                for chain in line.split(" "):
                    pdb = chain.split("_")[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def __iter__(self):
        # print(f"[DEBUG] Train sample")

        if self._sample_mode == "length_batch":
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby("modeled_seq_len").sample(
                self._batch_size, replace=True, random_state=self.epoch
            )
            return iter(sampled_order["index"].tolist())
        elif self._sample_mode == "time_batch":
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        elif self._sample_mode == "cluster_length_batch":
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv_group_clusters.sample(
                1, random_state=self.epoch
            )
            sampled_order = sampled_clusters.groupby("modeled_seq_len").sample(
                self._batch_size, replace=True, random_state=self.epoch
            )
            return iter(sampled_order["index"].tolist())
        elif self._sample_mode == "cluster_time_batch":
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv_group_clusters.sample(
                1, random_state=self.epoch
            )
            dataset_indices = sampled_clusters["index"].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        elif self._sample_mode == "cluster_time_batch_v2":
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv_group_clusters.sample(
                1, random_state=self.epoch
            )
            dataset_indices = sampled_clusters["index"].tolist()
            max_per_batch = sampled_clusters["max_batch_examples"].tolist()

            # Repeat each index to max batch size and pad until self._batch_size with None as indexes
            repeated_indices = []
            assert (
                self._batch_size % self._num_gpus == 0
            ), "Batch size must be divisible by num_gpus"

            # num_gpus = self._batch_size
            # setup_dataloaders(train_loader, use_distributed_sampler=False) Fixes actual batch
            # So we don't need this
            num_gpus = 1
            batch_size = self._batch_size // num_gpus

            for ix in range(num_gpus):
                for idx, count in zip(dataset_indices, max_per_batch):
                    # count = max(1, count // self._num_gpus)
                    # Repeat the index based on its count
                    repeated_indices += [idx] * min(count, batch_size)
                    repeated_indices += [None] * max(0, batch_size - count)

            return iter(repeated_indices)
        else:
            raise ValueError(f"Invalid sample mode: {self._sample_mode}")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len


class DistributedTrainSampler(TrainSampler):
    """
    Takes in a rank arg for shuffling
    """

    def __init__(
        self,
        *,
        data_conf,
        dataset,
        batch_size,
        sample_mode,
        rank,
        max_squared_res,
        num_gpus,
    ):
        self.rank = rank
        super().__init__(
            data_conf=data_conf,
            dataset=dataset,
            batch_size=batch_size,
            sample_mode=sample_mode,
            max_squared_res=max_squared_res,
            num_gpus=num_gpus,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch
        # self.epoch = epoch + 123456 * self.rank


# modified from torch.utils.data.distributed.DistributedSampler
# key points: shuffle of each __iter__ is determined by epoch num to ensure the same shuffle result for each proccessor
class OldDistributedTrainSampler(data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    modified from torch.utils.data.distributed import DistributedSampler

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        *,
        data_conf,
        dataset,
        batch_size,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv["index"] = self._dataset_indices
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._data_csv)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._batch_size)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate(
                    (
                        indices,
                        np.repeat(indices, math.ceil(padding_size / len(indices)))[
                            :padding_size
                        ],
                    ),
                    axis=0,
                )

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
