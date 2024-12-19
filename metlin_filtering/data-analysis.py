from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, MolToSmiles, PandasTools
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from tqdm.autonotebook import tqdm
from tqdm.contrib.concurrent import thread_map

from metlin_filtering import BASE_DIR, SEED, PROCESSED_DIR, OUTPUT_DIR, MODEL_NAMES
from metlin_filtering.preprocessing import load_processed_data
from metlin_filtering.training import merge_predictions


def generate_statistics(dataframe):
    dataframe["p_median"] = dataframe.loc[:, MODEL_NAMES].median(axis=1)
    dataframe["p_mean"] = dataframe.loc[:, MODEL_NAMES].mean(axis=1)
    dataframe["p_std"] = dataframe.loc[:, MODEL_NAMES].std(axis=1)
    dataframe["d_median"] = (dataframe.loc[:, "EXP"] -
                             dataframe.loc[:, "p_median"]).abs()
    dataframe["d_median_R"] = (dataframe["d_median"] /
                               dataframe.loc[:, "p_median"]).abs()

    dataframe["S_5"] = np.zeros(len(dataframe))
    dataframe["S_5_R"] = np.zeros(len(dataframe))
    dataframe["S_5_B"] = np.zeros(len(dataframe))
    for col in MODEL_NAMES:
        dataframe[f"d_{col}"] = (dataframe.loc[:, col] -
                                 dataframe.loc[:, "EXP"]).abs()
        dataframe[f"d_{col}_R"] = (
            dataframe.loc[:, f"d_{col}"] / dataframe.loc[:, "p_median"].abs())

        abs_mask = dataframe[f"d_{col}"] >= dataframe[f"d_{col}"].quantile(float(
            (100 - 5) / 100))
        rel_mask = dataframe[f"d_{col}_R"] >= (
            dataframe[f"d_{col}_R"].quantile(float((100 - 5) / 100)))

        dataframe.loc[abs_mask, "S_5"] += 1
        dataframe.loc[rel_mask, "S_5_R"] += 1
        dataframe.loc[abs_mask & rel_mask, "S_5_B"] += 1


def get_prediction_stats(dataframes):
    print("Prediction accuracy for the models averaged over the whole dataset.")
    prediction_stats = []
    for df in dataframes:
        prediction_stats.append(
            df[1].loc[:, [f"d_{x}" for x in MODEL_NAMES]].agg(["mean", "median"]))
        print(f"SEED {df[0]}", prediction_stats[-1])
    if len(prediction_stats) > 1:
        print(pd.concat(prediction_stats, axis=1).reset_index().melt(
            id_vars="index").groupby(["index", "variable"]).agg(["mean", "std"]))


def get_clean_prediction_stats(dataframes):
    print("Prediction accuracy for the models averaged over the dataset without \
          potentially erroneous retention times.")
    prediction_stats = []
    for df in dataframes:
        prediction_stats.append(
            df[1].loc[df[1]["S_5_B"] < 5, [f"d_{x}" for x in MODEL_NAMES]].agg(["mean", "median"]))
        print(f"SEED {df[0]}", prediction_stats[-1])
    if len(prediction_stats) > 1:
        print(pd.concat(prediction_stats, axis=1).reset_index().melt(
            id_vars="index").groupby(["index", "variable"]).agg(["mean", "std"]))


if __name__ == "__main__":
    predicted_files = OUTPUT_DIR.glob("predicted-smiles-*.csv")
    if not predicted_files:
        merge_predictions(SEED)
        predicted_files = OUTPUT_DIR.glob("predicted-smiles-*.csv")
    predicted_dfs = []
    for file in predicted_files:
        predicted_dfs.append(pd.read_csv(file, index_col=0))

    for df in predicted_dfs:
        generate_statistics(df)

    get_prediction_stats(predicted_dfs)
    get_clean_prediction_stats(predicted_dfs)
