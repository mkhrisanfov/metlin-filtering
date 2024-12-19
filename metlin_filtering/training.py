"""
training.py

Module to train and predict retention times using various machine learning models.

This module includes functions to clean the dataset, encode SMILES, generate fingerprints,
descriptors, and prepare datasets for different models. It also contains the main function
to perform 5-fold training and prediction using CNN, FCD, FCFP, GNN, and CatBoost models.

Functions:
    train_fn: Trains the model and evaluates it.
    main: Main function to execute the training and prediction process.

Classes:
    None
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor, Pool
from rdkit.Chem import MolToSmiles
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as GDataLoader
from tqdm.autonotebook import tqdm
from tqdm.contrib.concurrent import thread_map

from metlin_filtering.datasets import (CNN_Dataset, FCD_Dataset, FCFP_Dataset,
                                       get_gnn_dataset)
from metlin_filtering.models import CNN, FCD, FCFP, GNN
from metlin_filtering.preprocessing import get_clean_dataset, load_processed_data
from metlin_filtering.utils import (encode_smiles, generate_descriptors,
                                    generate_fingerprints)
from metlin_filtering import BASE_DIR, SEED, MODEL_NAMES, OUTPUT_DIR, PROCESSED_DIR


def train_fn(model, optim, loss_fn, epochs, train_dl, eval_dl, name):
    """
    Train a model and evaluate its performance.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    optim : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epochs : int
        The number of epochs to train for.
    train_dl : torch.utils.data.DataLoader
        The data loader for the training set.
    eval_dl : torch.utils.data.DataLoader
        The data loader for the evaluation set.
    name : str
        The name to use for saving the model.

    Returns
    -------
    b_train : float
        The best training loss observed during training.
    b_eval : float
        The best evaluation loss observed during training.
    """
    writer = SummaryWriter(log_dir=BASE_DIR / "logs" / name, flush_secs=15)

    torch.cuda.empty_cache()
    b_train = 1e5
    b_eval = 1e5

    bar = tqdm(range(epochs), leave=False)
    for epoch in bar:
        epoch_train_loss = model.train_fn(optim, loss_fn, train_dl)
        b_train = min(b_train, epoch_train_loss)

        epoch_eval_loss = model.eval_fn(loss_fn, eval_dl)
        b_eval = min(b_eval, epoch_eval_loss)

        bar.set_postfix_str(f"{epoch_train_loss:.3f} | {epoch_eval_loss:.3f}")

        writer.add_scalar("loss/train", epoch_train_loss, epoch)
        writer.add_scalar("loss/eval", epoch_eval_loss, epoch)
        if epoch_eval_loss <= b_eval:
            torch.save(model.state_dict(),
                       BASE_DIR / f"models/{name}_model-{SEED}.pth")
    torch.save(model.state_dict(),
               BASE_DIR / f"models/{name}_model_final-{SEED}.pth")
    bar.close()
    return b_train, b_eval


def merge_predictions(current_seed: int = SEED):
    print("Merging predictions")
    all_predictions = {}
    for name in MODEL_NAMES:
        predictions = []
        for file in PROCESSED_DIR.glob(f'5Fold-{name}_?_predictions-{current_seed}.npy'):
            predictions.append(np.load(file).ravel()*1000)
        all_predictions[name] = np.concatenate(predictions, axis=0)
    predicted_df = pd.DataFrame(all_predictions)

    input_inchi, input_molecules, input_retention_times = load_processed_data(
        PROCESSED_DIR / "clean_inchi_400_plus.csv")

    print("Preparing SMILES")
    input_smiles = np.array(thread_map(
        MolToSmiles, input_molecules, chunksize=500))

    unique_smiles, unique_index, unique_inverse = np.unique(
        input_smiles, return_index=True, return_inverse=True
    )

    kf = KFold(5, shuffle=True, random_state=current_seed)
    inp_retention_times = []
    inp_smiles = []
    for k, (unique_train_idx, unique_eval_idx) in enumerate(kf.split(
            np.arange(len(unique_smiles)))):
        input_eval_idx = unique_index[unique_eval_idx]

        eval_retention_times = input_retention_times[input_eval_idx]
        eval_smiles = input_smiles[input_eval_idx]

        inp_retention_times.append(eval_retention_times*1000)
        inp_smiles.append(eval_smiles)
    inp_retention_times = np.concatenate(inp_retention_times)
    inp_smiles = np.concatenate(inp_smiles)

    predicted_df["EXP"] = inp_retention_times
    predicted_df["SMILES"] = inp_smiles
    predicted_df.to_csv(OUTPUT_DIR / f"predicted-smiles-{current_seed}.csv")


def main(filename: str, train: bool = True, predict: bool = True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Cleaning METLIN from invalid entries")
    get_clean_dataset(filename)
    # Load data
    _, input_molecules, input_retention_times = load_processed_data(
        PROCESSED_DIR / "clean_inchi_400_plus.csv")

    print("Preparing SMILES")
    input_smiles = np.array(thread_map(
        MolToSmiles, input_molecules, chunksize=500))

    unique_smiles, unique_index, unique_inverse = np.unique(
        input_smiles, return_index=True, return_inverse=True
    )
    print("Full DS length", len(input_molecules))
    print("Unique DS length", len(unique_smiles))

    # CNN
    num_encoded_smiles, encoded_smiles = encode_smiles(input_smiles)

    # FCFP and CatBoost
    morgan_fingerprints, rdkit_fingerprints = generate_fingerprints(
        input_molecules)

    # FCD and CatBoost
    descriptors = generate_descriptors(input_molecules)

    # GNN
    gnn_num_fingerprints, gnn_full_dataset = get_gnn_dataset(
        molecules=input_molecules, retention_times=input_retention_times)

    # KFold split
    if train:
        print("Begin 5-Fold training")
        print("Model\t\t Iteration\t Train Loss\t Eval Loss")
    elif predict:
        print("Begin 5-Fold prediction")
    kf = KFold(5, shuffle=True, random_state=SEED)
    for k, (unique_train_idx, unique_eval_idx) in enumerate(kf.split(
            np.arange(len(unique_smiles)))):

        input_train_idx = unique_index[unique_train_idx]
        input_eval_idx = unique_index[unique_eval_idx]

        if train:
            train_retention_times = input_retention_times[input_train_idx]
            train_encoded_smiles = encoded_smiles[input_train_idx]
            train_descriptors = descriptors[input_train_idx]
            train_morgan_fingerprints = morgan_fingerprints[input_train_idx]
            train_rdkit_fingerprints = rdkit_fingerprints[input_train_idx]
            train_gnn_dataset = [gnn_full_dataset[i] for i in input_train_idx]

        eval_retention_times = input_retention_times[input_eval_idx]
        eval_descriptors = descriptors[input_eval_idx]
        eval_morgan_fingerprints = morgan_fingerprints[input_eval_idx]
        eval_rdkit_fingerprints = rdkit_fingerprints[input_eval_idx]
        eval_encoded_smiles = encoded_smiles[input_eval_idx]
        eval_gnn_dataset = [gnn_full_dataset[i] for i in input_eval_idx]

        # CNN
        model = CNN(num_encoded_smiles).to(device)

        eval_ds = CNN_Dataset(eval_encoded_smiles,
                              eval_retention_times)
        eval_dl = DataLoader(eval_ds,
                             batch_size=512,
                             pin_memory=True,
                             shuffle=False,
                             num_workers=4,
                             persistent_workers=True)
        if train:
            train_ds = CNN_Dataset(train_encoded_smiles,
                                   train_retention_times)
            train_dl = DataLoader(train_ds,
                                  batch_size=512,
                                  pin_memory=True,
                                  shuffle=True,
                                  num_workers=4,
                                  persistent_workers=True)
            b_train, b_eval = train_fn(model=model,
                                       optim=torch.optim.Adam(
                                           model.parameters(),
                                           lr=3e-4),
                                       loss_fn=nn.L1Loss(reduction='sum'),
                                       epochs=200,
                                       train_dl=train_dl,
                                       eval_dl=eval_dl,
                                       name=f"5Fold-CNN_{k}")
            print(f"5Fold-CNN\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        if predict:
            model.load_state_dict(torch.load(
                BASE_DIR / f"models/5Fold-CNN_{k}_model-{SEED}.pth",
                map_location=device, weights_only=True))
            predicted_retention_times = model.eval_fn(
                nn.L1Loss(reduction='sum'), eval_dl, return_predictions=True)
            np.save(
                PROCESSED_DIR / f"5Fold-CNN_{k}_predictions-{SEED}.npy",
                predicted_retention_times)

        # FCD
        model = FCD().to(device)

        eval_ds = FCD_Dataset(eval_descriptors, eval_retention_times)
        eval_dl = DataLoader(eval_ds,
                             batch_size=512,
                             pin_memory=True,
                             shuffle=False,
                             num_workers=4,
                             persistent_workers=True)
        if train:
            train_ds = FCD_Dataset(train_descriptors, train_retention_times)
            train_dl = DataLoader(train_ds,
                                  batch_size=512,
                                  pin_memory=True,
                                  shuffle=True,
                                  num_workers=4,
                                  persistent_workers=True)
            b_train, b_eval = train_fn(model=model,
                                       optim=torch.optim.Adam(
                                           model.parameters(),
                                           lr=3e-4),
                                       loss_fn=nn.L1Loss(reduction='sum'),
                                       epochs=150,
                                       train_dl=train_dl,
                                       eval_dl=eval_dl,
                                       name=f"5Fold-FCD_{k}")
            print(f"5Fold-FCD\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")

        if predict:
            model.load_state_dict(torch.load(
                BASE_DIR / f"models/5Fold-FCD_{k}_model-{SEED}.pth",
                map_location=device, weights_only=True))
            predicted_retention_times = model.eval_fn(
                nn.L1Loss(reduction='sum'), eval_dl, return_predictions=True)
            np.save(
                PROCESSED_DIR / f"5Fold-FCD_{k}_predictions-{SEED}.npy",
                predicted_retention_times)

        # FCFP
        model = FCFP().to(device)
        eval_ds = FCFP_Dataset(eval_rdkit_fingerprints,
                               eval_morgan_fingerprints,
                               eval_retention_times)
        eval_dl = DataLoader(eval_ds,
                             batch_size=512,
                             pin_memory=True,
                             shuffle=False,
                             num_workers=4,
                             persistent_workers=True)
        if train:
            train_ds = FCFP_Dataset(train_rdkit_fingerprints,
                                    train_morgan_fingerprints,
                                    train_retention_times)
            train_dl = DataLoader(train_ds,
                                  batch_size=512,
                                  pin_memory=True,
                                  shuffle=True,
                                  num_workers=4,
                                  persistent_workers=True)

            b_train, b_eval = train_fn(model=model,
                                       optim=torch.optim.Adam(
                                           model.parameters(),
                                           lr=3e-4),
                                       loss_fn=nn.L1Loss(reduction='sum'),
                                       epochs=150,
                                       train_dl=train_dl,
                                       eval_dl=eval_dl,
                                       name=f"5Fold-FCFP_{k}")
            print(f"5Fold-FCFP\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        if predict:
            model.load_state_dict(torch.load(
                BASE_DIR / f"models/5Fold-FCFP_{k}_model-{SEED}.pth",
                map_location=device, weights_only=True))
            predicted_retention_times = model.eval_fn(
                nn.L1Loss(reduction='sum'), eval_dl, return_predictions=True)
            np.save(
                PROCESSED_DIR / f"5Fold-FCFP_{k}_predictions-{SEED}.npy",
                predicted_retention_times)

        # GNN
        model = GNN(gnn_num_fingerprints).to(device)
        eval_dl = GDataLoader(eval_gnn_dataset,
                              batch_size=128,
                              pin_memory=True,
                              shuffle=False)
        if train:
            train_dl = GDataLoader(train_gnn_dataset,
                                   batch_size=128,
                                   shuffle=True,
                                   pin_memory=True)
            b_train, b_eval = train_fn(model=model,
                                       optim=torch.optim.Adam(
                                           model.parameters(),
                                           lr=3e-4),
                                       loss_fn=nn.L1Loss(reduction='sum'),
                                       epochs=150,
                                       train_dl=train_dl,
                                       eval_dl=eval_dl,
                                       name=f"5Fold-GNN_{k}")
            print(f"5Fold-GNN\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        if predict:
            model.load_state_dict(torch.load(
                BASE_DIR / f"models/5Fold-GNN_{k}_model-{SEED}.pth",
                map_location=device, weights_only=True))
            predicted_retention_times = model.eval_fn(
                nn.L1Loss(reduction='sum'), eval_dl, return_predictions=True)
            np.save(
                PROCESSED_DIR / f"5Fold-GNN_{k}_predictions-{SEED}.npy",
                predicted_retention_times)

        # CatBoost
        eval_ds = Pool(np.hstack([
            eval_rdkit_fingerprints,
            eval_morgan_fingerprints,
            eval_descriptors
        ]),
            eval_retention_times)
        if train:
            trn_ds = Pool(np.hstack([
                train_rdkit_fingerprints,
                train_morgan_fingerprints,
                train_descriptors
            ]),
                train_retention_times)

            model = CatBoostRegressor(
                loss_function="MAE",
                task_type="GPU",
                devices="0",
                metric_period=20,
                learning_rate=5e-3,
                depth=10,
                iterations=5000,
                use_best_model=True,
                best_model_min_trees=4000,
                silent=True,
                allow_writing_files=False
            )
            model.fit(
                trn_ds, eval_set=eval_ds, plot=False)
            model.save_model(BASE_DIR / f"models/5Fold-CB_{k}_model-{SEED}")
            b_train = model.get_best_score().get("learn",
                                                 {"MAE": np.nan})["MAE"]
            b_eval = model.get_best_score().get("validation",
                                                {"MAE": np.nan})["MAE"]
            print(f"5Fold-CB\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        if predict:
            model = CatBoostRegressor().load_model(
                BASE_DIR / f"models/5Fold-CB_{k}_model-{SEED}")
            predicted_retention_times = np.stack(
                model.predict(eval_ds))
            np.save(
                PROCESSED_DIR / f"5Fold-CB_{k}_predictions-{SEED}.npy",
                predicted_retention_times)


if __name__ == "__main__":
    main(filename=BASE_DIR / "data/input/SMRT_dataset.csv",
         train=True, predict=True)
    merge_predictions(SEED)
