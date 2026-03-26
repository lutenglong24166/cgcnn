from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

TORCH_DTYPE = torch.float32


def build_atom_feature(structure: Structure):
    # atom_feature shape is (N,)
    # values are 0-based atomic-number indices for nn.Embedding
    atom_feature = torch.tensor(
        [site.specie.number - 1 for site in structure],
        dtype=torch.long,
    )

    return atom_feature


def build_edge_feature(structure: Structure, radius: float, dmin: float, step: float):
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)

    src, dst, dist = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        for n in nbrs:
            j = n[2]  # neighbor atom index
            d = n[1]  # distance
            src.append(i)
            dst.append(j)
            dist.append(d)

    # edge_index shape is (E, 2)
    # E is total number of neighbors
    # 2 is for ids of source and destination atoms
    edge_index = torch.LongTensor(np.stack([src, dst], axis=1))

    gde = GaussianDistanceExpander(dmin, radius, step)
    # edge_feature shape is (E, B)
    # B is the number of bases, determined by number of bases.
    edge_feature = torch.tensor(gde.expand(dist), dtype=TORCH_DTYPE)

    return edge_index, edge_feature


@dataclass
class GaussianDistanceExpander:
    dmin: float
    dmax: float
    step: float

    def __post_init__(self):
        self.bases = np.arange(self.dmin, self.dmax + self.step, self.step)
        self.var = self.step

    def expand(self, distances: list | np.ndarray):
        if not isinstance(distances, np.ndarray):
            # an array with the lenghth of N
            distances = np.array(distances)

        # shape is (N, M)
        # M is the number of bases
        feature = np.exp(
            -((distances[..., np.newaxis] - self.bases) ** 2) / self.var**2
        )

        return feature


class StructureData(Dataset):
    def __init__(
        self,
        data_fn,
        radius: float = 5.0,
        dmin: float = 0.0,
        step: float = 0.2,
    ):
        super().__init__()

        data = loadfn(data_fn)
        self.sample_ids = data["sample_ids"]
        self.structures = data["structures"]
        self.targets = data["targets"]

        self.radius = radius
        self.dmin = dmin
        self.step = step

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        structure = self.structures[idx]

        atom_feature = build_atom_feature(structure)
        edge_index, edge_feature = build_edge_feature(
            structure, self.radius, self.dmin, self.step
        )

        target = self.targets[idx]
        target = torch.tensor([float(target)], dtype=TORCH_DTYPE)

        return (atom_feature, edge_feature, edge_index), target, sample_id


def collate_data(batch_data: list):
    """
    Collate of a list of dataset into batch data.

    batch_data: list of
      ((atom_feature, edge_feature, edge_index), target, sample_id)

    atom_feature:   [N_i]
    edge_feature:   [E_i, B]
    edge_index: [E_i, 2]
    target:     [1]
    """

    batch_atom_feature = []
    batch_edge_feature = []
    batch_edge_index = []
    crystal_atom_idx = []
    batch_target = []
    batch_ids = []

    base_idx = 0
    for (atom_feature, edge_feature, edge_index), target, sample_id in batch_data:
        n_i = atom_feature.size()[0]  # number of atom in i_th structure

        # add elements to batched atom_feature list
        batch_atom_feature.append(atom_feature)

        # record the global atom indices for this crystal
        crystal_atom_idx.append(torch.arange(n_i, dtype=torch.long) + base_idx)

        if edge_index.numel() > 0:  # check if there is edges in the crystal
            # add elements to batched edge_index and edge_feature list
            batch_edge_index.append(edge_index + base_idx)
            batch_edge_feature.append(edge_feature)

        # add elements to batched target and sample_id lists
        batch_target.append(target)
        batch_ids.append(sample_id)

        base_idx += n_i

    # concatenate atom_feature
    # (N_total,)
    batch_atom_feature = torch.cat(batch_atom_feature, dim=0)

    # concatenate edge_featrue and edge_index
    # batch_edge_feature: (E_total, B)
    # batch_edge_index: (E_total, 2)
    batch_edge_feature = torch.cat(batch_edge_feature, dim=0)
    batch_edge_index = torch.cat(batch_edge_index, dim=0)

    # stack target
    # (batch_size, 1)
    batch_target = torch.stack(batch_target, dim=0)

    return (
        (batch_atom_feature, batch_edge_feature, batch_edge_index, crystal_atom_idx),
        batch_target,
        batch_ids,
    )


def get_train_val_test_loader(
    dataset: Dataset,
    collate_fn: Callable,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
):
    total_size = len(dataset)

    indices = np.arange(total_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size : train_size + val_size].tolist()
    test_idx = indices[
        train_size + val_size : train_size + val_size + test_size
    ].tolist()

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=SubsetRandomSampler(test_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
