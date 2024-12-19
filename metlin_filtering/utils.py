"""
utils.py

This module provides utility functions for encoding SMILES strings,
generating molecular fingerprints, loading and processing data,
and generating molecular descriptors.
It is designed to be used with RDKit for molecular validation
and fingerprint generation, and Pandas for data manipulation.

Functions:
    encode_smiles(smiles): Encodes a list of SMILES strings into numerical format.
    generate_fingerprints(molecules): Generates fingerprints for a list of molecules.
    load_processed_data(input_file_name): Loads and processes the input data.
    generate_descriptors(molecules): Generates descriptors for a list of molecules.

"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm
from tqdm.contrib.concurrent import thread_map
from metlin_filtering import BASE_DIR


def get_ecfp_from_smiles(smiles, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=1024)
    fp_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def get_maccs_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_ecfp_from_inchi(inchi, radius: int = 2):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=1024)
    fp_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def get_maccs_from_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_maccs_from_mol(mol):
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_smiles_from_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=False)


def encode_smiles(smiles):
    """
    Encode a list of SMILES strings into numerical format.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings to encode.

    Returns
    -------
    encoded_smiles : torch.Tensor
        A tensor of encoded SMILES strings.
    """
    sym_dict = defaultdict(lambda: len(sym_dict))

    encoded_smiles = torch.zeros((len(smiles), 256), dtype=torch.long)
    for s, smiles in enumerate(tqdm(smiles)):
        if "i" in smiles or "l" in smiles or "r" in smiles:
            smiles.replace("Si", "X").replace("Cl", "Y").replace("Br", "Z")
        for s2, sym in enumerate(smiles):
            encoded_smiles[s, s2] = sym_dict[sym]
    return len(sym_dict), encoded_smiles


def generate_fingerprints(molecules):
    """
    Generate fingerprints for a list of molecules.

    Parameters
    ----------
    molecules : list of rdkit.rdChem.Mol
        A list of molecules to generate fingerprints for.

    Returns
    -------
    morgan_fingerprints : numpy.ndarray
        An array of Morgan fingerprints for the input molecules.
    rdkit_fingerprints : numpy.ndarray
        An array of RDKit fingerprints for the input molecules.
    """
    # FCFP and CatBoost
    print("Generating Morgan fingerprints")
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=1024)
    morgan_fingerprints = np.array(thread_map(
        morgan_generator.GetCountFingerprintAsNumPy, molecules, chunksize=500))

    # FCFP and CatBoost
    print("Generating RDKit fingerprints")
    rdkit_generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
    rdkit_fingerprints = np.array(thread_map(
        rdkit_generator.GetCountFingerprintAsNumPy, molecules, chunksize=500))
    return morgan_fingerprints, rdkit_fingerprints


def generate_descriptors(molecules):
    """
    Generate descriptors for a list of molecules.

    Parameters
    ----------
    molecules : list of rdkit.rdChem.Mol
        A list of molecules to generate descriptors for.

    Returns
    -------
    descriptors : numpy.ndarray
        An array of RDKit descriptors for the input molecules.
    """
    print("Loading / Generating RDKit descriptors")
    if not os.path.exists(BASE_DIR / "data" / "processed" / "cl_400+_desriptors.npy"):
        descriptors = thread_map(CalcMolDescriptors,
                                 molecules, chunksize=500)
        pd_descriptors = pd.DataFrame(descriptors)
        print(pd_descriptors.shape)
        np.save(BASE_DIR / "data" / "processed" / "cl_400+_desriptors.npy",
                pd_descriptors.to_numpy(dtype=float, na_value=0))
    else:
        descriptors = np.load(
            BASE_DIR / "data" / "processed" / "cl_400+_desriptors.npy")

    print("Scaling RDKit descriptors to zero mean and unit variance")
    scl = StandardScaler()
    descriptors = scl.fit_transform(descriptors)
    return descriptors
