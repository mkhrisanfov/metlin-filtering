"""
preprocessing.py

This module contains functions for preprocessing METLIN SMRT molecular data from a CSV file.
It includes functions to filter entries based on retention time, validate InChI strings using RDKit,
and save the cleaned dataset to a new CSV file.
The module can be executed as a standalone script.

Functions:
    get_clean_dataset(filename):
        Filters entries based on retention time, validates InChI strings using RDKit,
        and saves the cleaned dataset.

Attributes:
    BASE_DIR (pathlib.Path): The base directory of the project.

"""

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import thread_map
from rdkit import Chem
from metlin_filtering import BASE_DIR


def load_processed_data(input_file_name):
    """
    Load and process the input data.

    Parameters
    ----------
    input_file_name : str
        The path to the input file.

    Returns
    -------
    input_inchi : numpy.ndarray
        The InChI for the input molecules.
    input_molecules : numpy.ndarray
        The input molecules.
    input_retention_times : numpy.ndarray
        The input retention times.
    """
    input_df = pd.read_csv(input_file_name, index_col=0)
    input_inchi = input_df["inchi"].to_numpy()
    input_retention_times = input_df["rt"].to_numpy()/1000
    print("Generating molecules")
    input_molecules = np.array(thread_map(
        Chem.MolFromInchi, input_inchi, chunksize=500))
    return input_inchi, input_molecules, input_retention_times


def get_clean_dataset(filename):
    input_df = pd.read_csv(filename, sep=";")

    print("Filtering by retention time > 400s")
    filtered_df = input_df[input_df["rt"] > 400]
    filtered_inchi = filtered_df["inchi"].to_numpy()
    print(f"# entries remain: {len(filtered_df)}")

    print("Creating molecules with RDKit")
    mols = thread_map(Chem.MolFromInchi, filtered_df["inchi"].to_numpy())
    bad_inchi = [filtered_inchi[i]
                 for i, mol in enumerate(mols) if mol is None]
    print(f"# of invalid entries {len(bad_inchi)}")
    print(bad_inchi)

    clean_inchi_mask = [True if mol else False for mol in mols]
    print(f"# entries remain: {np.count_nonzero(clean_inchi_mask)}")
    clean_df = filtered_df[clean_inchi_mask]
    print("Saving prefiltered METLIN to ")
    clean_df.to_csv(
        BASE_DIR / "data" / "processed" / "clean_inchi_400_plus.csv")


if __name__ == "__main__":
    get_clean_dataset(BASE_DIR / "data" / "input" / "SMRT_dataset.csv")
