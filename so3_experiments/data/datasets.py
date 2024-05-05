"""Copyright (c) Dreamfold."""
from torch.utils.data import Dataset
import numpy as np
import torch
import os


def get_split(data, split, seed):
    assert split in ["train", "valid", "test", "all"], f"split {split} not supported."
    if split != "all":
        rng = np.random.default_rng(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)

        n = len(data)
        if split == "train":
            data = data[indices[: int(n * 0.8)]]
        elif split == "valid":
            data = data[indices[int(n * 0.8) : int(n * 0.9)]]
        elif split == "test":
            data = data[indices[int(n * 0.9) :]]
    return data


class SpecialOrthogonalGroup(Dataset):
    def __init__(self, root="data", split="train", seed=12345):
        data = np.load(f"{root}/orthogonal_group.npy").astype("float32")
        self.data = get_split(data, split, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
