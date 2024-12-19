"""
models.py

This module contains the definitions of different neural network architectures
used for predicting retention times based on molecular information.
Each model includes forward methods for computation,
and train_fn and eval_fn methods for training and evaluation, respectively.

Classes:
    CNN: Convolutional Neural Network for processing molecular encodings.
    FCD: Fully Connected Network for processing molecular descriptors.
    FCFP: Fully Connected Network for processing molecular fingerprints.
    GNN: Graph Neural Network for processing molecular graphs.

Functions:
    None

Attributes:
    None
"""


import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class CNN(nn.Module):

    def __init__(self, n_sym):
        super().__init__()
        self.embed = nn.Embedding(n_sym, 32)
        self.conv_in = nn.Sequential(
            nn.Conv1d(32, 256, 9, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 256, 9, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.lin0 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Identity(),
        )

    def forward(self, x):
        x = torch.permute(self.embed(x), (0, 2, 1))
        x = self.conv_in(x)
        x = self.lin0(torch.mean(x, dim=2))
        return self.out(x).squeeze()

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for encoded_smiles, retention_times in train_dl:
            optim.zero_grad()
            pred_retention_times = self.forward(
                encoded_smiles.to(next(self.parameters()).device, non_blocking=True))
            loss = loss_fn(
                pred_retention_times,
                retention_times.to(next(self.parameters()).device, non_blocking=True))
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_retention_times = []
        with torch.no_grad():
            for encoded_smiles, retention_times in eval_dl:
                pred_retention_times = self.forward(
                    encoded_smiles.to(next(self.parameters()).device, non_blocking=True))
                if return_predictions:
                    prediction_retention_times.append(pred_retention_times)
                loss = loss_fn(
                    pred_retention_times,
                    retention_times.to(next(self.parameters()).device, non_blocking=True))
                epoch_eval_loss += loss
        if return_predictions:
            return torch.cat(prediction_retention_times, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)


class FCD(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_in = nn.Sequential(
            nn.Linear(210, 2048),
            nn.SiLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Identity(),
        )

    def forward(self, descriptors):
        x = self.fc_in(descriptors)
        y = self.fc1(x)
        z = self.fc2(x+y)
        x = self.fc3(z+y)
        return self.fc_out(x+z)

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for descriptors, retention_times in train_dl:
            optim.zero_grad()
            pred_retention_times = self.forward(
                descriptors.to(next(self.parameters()).device, non_blocking=True))
            loss = loss_fn(
                pred_retention_times,
                retention_times.to(next(self.parameters()).device, non_blocking=True))
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_retention_times = []
        with torch.no_grad():
            for descriptors, retention_times in eval_dl:
                pred_retention_times = self.forward(
                    descriptors.to(next(self.parameters()).device, non_blocking=True))
                if return_predictions:
                    prediction_retention_times.append(pred_retention_times)
                loss = loss_fn(
                    pred_retention_times,
                    retention_times.to(next(self.parameters()).device, non_blocking=True))
                epoch_eval_loss += loss
        if return_predictions:
            return torch.cat(prediction_retention_times, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)


class FCFP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_in = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.SiLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.SiLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.SiLU(inplace=True),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Identity(),
        )

    def forward(self, rdkit_fingerprints, morgan_fingerprints):
        x = self.fc_in(
            torch.cat([rdkit_fingerprints, morgan_fingerprints], dim=1))
        x1 = self.fc1(x)
        x2 = self.fc2(x+x1)
        return self.fc_out(x1+x2)

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for rdkit_fingerprints, morgan_fingerprints, retention_times in train_dl:
            optim.zero_grad()
            pred_retention_times = self.forward(
                rdkit_fingerprints.to(
                    next(self.parameters()).device, non_blocking=True),
                morgan_fingerprints.to(
                    next(self.parameters()).device, non_blocking=True))
            loss = loss_fn(
                pred_retention_times,
                retention_times.to(next(self.parameters()).device, non_blocking=True))
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_retention_times = []
        with torch.no_grad():
            for rdkit_fingerprints, morgan_fingerprints, retention_times in eval_dl:
                pred_retention_times = self.forward(
                    rdkit_fingerprints.to(
                        next(self.parameters()).device, non_blocking=True),
                    morgan_fingerprints.to(
                        next(self.parameters()).device, non_blocking=True))
                if return_predictions:
                    prediction_retention_times.append(pred_retention_times)
                loss = loss_fn(
                    pred_retention_times,
                    retention_times.to(next(self.parameters()).device, non_blocking=True))
                epoch_eval_loss += loss
        if return_predictions:
            return torch.cat(prediction_retention_times, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)


class GNN(nn.Module):
    def __init__(self, n_fps):
        super().__init__()
        self.embed = nn.Embedding(n_fps, 64)
        self.in_conv = self._make_layer(64, 128)
        self.conv_layers = nn.ModuleList(
            [self._make_layer(128, 128) for _ in range(5)])
        self.lin_layers = nn.ModuleList(
            [self._make_lin_layer(256) for _ in range(4)])
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Identity()
        )

    def _make_layer(self, in_channels, out_channels):
        return gnn.Sequential(
            'x, edge_index', [
                (gnn.GraphConv(in_channels, out_channels), 'x, edge_index -> x'),
                (nn.LeakyReLU(inplace=True), 'x -> x'),
                (gnn.GraphConv(out_channels, out_channels), 'x, edge_index -> x'),
                (nn.LeakyReLU(inplace=True), 'x -> x'),
            ]
        )

    def _make_lin_layer(self, hidden):
        return nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embed(x)

        x = self.in_conv(x, edge_index)
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                y = conv(x, edge_index)
            elif i % 2 == 0:
                y = conv(x + y, edge_index)
            else:
                x = conv(x + y, edge_index)

        x = torch.cat([
            gnn.pool.global_mean_pool(x, data.batch),
            gnn.pool.global_mean_pool(y, data.batch)], dim=1)

        for i, lin in enumerate(self.lin_layers):
            if i == 0:
                y = lin(x)
            elif i % 2 == 0:
                y = lin(x + y)
            else:
                x = lin(x + y)
        return self.out(x)

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for data in train_dl:
            optim.zero_grad()
            pred_retention_times = self.forward(
                data.to(next(self.parameters()).device))
            loss = loss_fn(
                pred_retention_times.squeeze(),
                data.y)
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_retention_times = []
        with torch.no_grad():
            for data in eval_dl:
                pred_retention_times = self.forward(
                    data.to(next(self.parameters()).device))
                if return_predictions:
                    prediction_retention_times.append(pred_retention_times)
                loss = loss_fn(
                    pred_retention_times.squeeze(),
                    data.y)
                epoch_eval_loss += loss
        if return_predictions:
            return torch.cat(prediction_retention_times, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)
