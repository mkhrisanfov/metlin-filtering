"""
datasets.py
-----------

This module contains classes and functions for creating and handling
different types of datasets used in the filtering metlin project. These datasets
are primarily used for training and evaluating various machine learning models
in the context of HPLC retention time prediction.

Classes:
    CNN_Dataset: A custom Dataset class for handling encoded SMILES strings
                and their corresponding retention times.
    FCD_Dataset: A custom Dataset class for handling molecular descriptors
                and their corresponding retention times.
    FCFP_Dataset: A custom Dataset class for handling molecular fingerprints
                (ECFP and RDKit) and their corresponding retention times.
    Data: A class from torch_geometric.data used to represent graph-structured data.

Functions:
    get_gnn_dataset: A function to create a graph-structured dataset
                    for Graph Neural Network (GNN) models, using RDKit molecules
                    and their corresponding retention times.
"""

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data


class CNN_Dataset(Dataset):
    def __init__(self, encoded_smis, retention_times):
        self.encoded_smis = torch.LongTensor(encoded_smis)
        self.retention_times = torch.FloatTensor(retention_times)

    def __getitem__(self, index):
        return (self.encoded_smis[index], self.retention_times[index])

    def __len__(self):
        return len(self.retention_times)


class FCD_Dataset(Dataset):
    def __init__(self, descriptors, retention_times):
        retention_times = np.vstack(retention_times)
        self.retention_times = torch.FloatTensor(retention_times)
        self.descriptors = torch.FloatTensor(descriptors)

    def __getitem__(self, index):
        return (self.descriptors[index], self.retention_times[index])

    def __len__(self):
        return len(self.retention_times)


class FCFP_Dataset(Dataset):
    def __init__(self, rdkit_fingerprints, morgan_fingerprints, retention_times):
        retention_times = np.vstack(retention_times)
        self.retention_times = torch.FloatTensor(retention_times)

        self.rdkit_fingerprints = torch.FloatTensor(
            np.vstack(rdkit_fingerprints))
        self.morgan_fingerprints = torch.FloatTensor(
            np.vstack(morgan_fingerprints))

    def __getitem__(self, index):
        return (self.rdkit_fingerprints[index],
                self.morgan_fingerprints[index],
                self.retention_times[index])

    def __len__(self):
        return len(self.retention_times)


def get_gnn_dataset(molecules, retention_times):
    print("Preparing GNN Dataset")
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    hybdn_dict = defaultdict(lambda: len(hybdn_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

    gnn_dataset = []
    for m, mol in enumerate(tqdm(molecules)):
        fingerprints = np.zeros(mol.GetNumAtoms())
        for a, atom in enumerate(mol.GetAtoms()):
            atom_str = atom.GetSmarts()
            neighbors_str = "".join(
                sorted([ngh.GetSymbol() for ngh in atom.GetNeighbors()]))
            fingerprints[a] = fingerprint_dict[f"{atom_str}_{neighbors_str}"]
        mol_bonds = np.zeros((2, 2*mol.GetNumBonds()))
        for b, bond in enumerate(mol.GetBonds()):
            mol_bonds[:, b] = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            mol_bonds[:, -b-1] = bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()

        gnn_dataset.append(Data(
            x=torch.LongTensor(fingerprints),
            edge_index=torch.LongTensor(mol_bonds),
            y=retention_times[m],
        ))
    print("Atom Dict", len(atom_dict))
    print("Hybdn Dict", len(hybdn_dict))
    print("FPs Dict", len(fingerprint_dict))
    num_fingerprints = len(fingerprint_dict)
    return num_fingerprints, gnn_dataset
