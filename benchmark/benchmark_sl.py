import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.functional as F
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from fm4g.constraints import (compute_generation_cost,
                              compute_line_limit_violation,
                              compute_power_flow_violation)
from fm4g.models.hgnn import HGT, HeteroGNN
from fm4g.opf import OPFDataset, process_matpower_file
from utils import set_seed, parse_args


class HeteroGNNLightning(pl.LightningModule):
    def __init__(self, model, lr=0.001, matpower_data=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.matpower_data = matpower_data
        self.register_buffer("Y_real", torch.tensor(matpower_data.Y.real.todense(), dtype=torch.float32))
        self.register_buffer("Y_imag", torch.tensor(matpower_data.Y.imag.todense(), dtype=torch.float32))
        self.register_buffer("load_bus_indices", torch.tensor(matpower_data.load_bus_indices, dtype=torch.long))

    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)

    def training_step(self, batch, batch_idx):
        out = self(batch.x_dict, batch.edge_index_dict)
        vm_pred = out['bus'][:, 0] * (batch['bus'].x[:, 3] - batch['bus'].x[:, 2]) + batch['bus'].x[:, 2]
        va_pred = out['bus'][:, 1] * 360 - 180
        pg_pred = out['generator'][:, 0] * (batch['generator'].x[:, 3] - batch['generator'].x[:, 2]) + \
            batch['generator'].x[:, 2]
        qg_pred = out['generator'][:, 1] * (batch['generator'].x[:, 6] - batch['generator'].x[:, 5]) + \
            batch['generator'].x[:, 5]

        bus_mse = F.mse_loss(torch.stack([vm_pred, va_pred], dim=1), batch['bus'].y)
        gen_mse = F.mse_loss(torch.stack([pg_pred, qg_pred], dim=1), batch['generator'].y)

        loss = bus_mse + gen_mse
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x_dict, batch.edge_index_dict)
        vmin = batch['bus'].x[:, 2]
        vmax = batch['bus'].x[:, 3]
        vm_pred = out['bus'][:, 0] * (vmax - vmin) + vmin
        va_pred = out['bus'][:, 1] * 360 - 180

        pmin = batch['generator'].x[:, 2]
        pmax = batch['generator'].x[:, 3]
        qmin = batch['generator'].x[:, 5]
        qmax = batch['generator'].x[:, 6]
        pg_pred = out['generator'][:, 0] * (pmax - pmin) + pmin
        qg_pred = out['generator'][:, 1] * (qmax - qmin) + qmin

        bus_mse = F.mse_loss(torch.stack([vm_pred, va_pred], dim=1), batch['bus'].y)
        gen_mse = F.mse_loss(torch.stack([pg_pred, qg_pred], dim=1), batch['generator'].y)

        base_mva = batch['baseMVA'].item()
        pg_pred_pu = pg_pred / base_mva
        qg_pred_pu = qg_pred / base_mva
        pd = batch['load'].x[:, 0] / base_mva
        qd = batch['load'].x[:, 1] / base_mva

        # Power injections
        # NOTE: align the shape of pd with bus vm_pred
        P_inj = torch.zeros_like(vm_pred, dtype=torch.float32, device=vm_pred.device)
        P_inj[self.load_bus_indices] = -pd
        Q_inj = torch.zeros_like(vm_pred, dtype=torch.float32, device=vm_pred.device)
        Q_inj[self.load_bus_indices] = -qd

        # P_inj = -pd
        # Q_inj = -qd
        P_inj[batch['bus', 'generator_link', 'generator'].edge_index[0, :]] += pg_pred_pu
        Q_inj[batch['bus', 'generator_link', 'generator'].edge_index[0, :]] += qg_pred_pu

        rate_a = batch['bus', 'ac_line', 'bus'].edge_attr[:, 6]
        gencost_coeff = batch['generator'].x[:, -3:]
        edge_index = batch['ac_line'].edge_index

        bus_loss = compute_power_flow_violation(vm_pred,
                                                va_pred,
                                                P_inj,
                                                Q_inj,
                                                self.Y_real,
                                                self.Y_imag)
        gen_loss = compute_generation_cost(gencost_coeff, pg_pred)
        line_loss = compute_line_limit_violation(vm_pred, va_pred, self.Y_real, self.Y_imag, rate_a, edge_index)

        loss = bus_mse + gen_mse + gen_loss
        self.log('val_loss', loss)
        self.log('phy_loss', bus_loss + line_loss + gen_loss)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("config.yaml", "r") as config_file:
        env_config = yaml.safe_load(config_file)

    file_path = "./data/pglib/"
    matpower_instances = {
        "case14_ieee": "pglib_opf_case14_ieee.m",
        "case30_ieee": "pglib_opf_case30_ieee.m",
        "case118_ieee": "pglib_opf_case118_ieee.m",
        "case2000_goc": "pglib_opf_case2000_goc.m",
    }

    data = {}
    load, Y_real, Y_imag, base_mva = {}, {}, {}, {}
    vmax, vmin, qmax, qmin, pmax, pmin = {}, {}, {}, {}, {}, {}
    gencost, rate_a = {}, {}

    for case_name, case_file in matpower_instances.items():
        data[case_name] = process_matpower_file(file_path + case_file).to(device)
        load[case_name] = data[case_name]['load'].x.clone()
        Y_real[case_name] = torch.tensor(data[case_name].Y.real.todense(), dtype=torch.float32, device=device)
        Y_imag[case_name] = torch.tensor(data[case_name].Y.imag.todense(), dtype=torch.float32, device=device)
        base_mva[case_name] = data[case_name].baseMVA
        vmax[case_name] = data[case_name]['bus'].x[:, 2]
        vmin[case_name] = data[case_name]['bus'].x[:, 3]
        qmax[case_name] = data[case_name]['generator'].x[:, 0]
        qmin[case_name] = data[case_name]['generator'].x[:, 1]
        pmax[case_name] = data[case_name]['generator'].x[:, 4]
        pmin[case_name] = data[case_name]['generator'].x[:, 5]
        gencost[case_name] = data[case_name]['gencost'].x
        rate_a[case_name] = data[case_name]['bus', 'ac_line', 'bus'].edge_attr[:, 3]

    data_keys = list(data.keys())
    train_data_keys = test_data_keys = data_keys[:3]
    test_data_keys = data_keys[3:]

    metadata = data[data_keys[0]].metadata()
    input_channels = {
        'bus': data[data_keys[0]]['bus'].x.size(1),
        'generator': data[data_keys[0]]['generator'].x.size(1),
        'gencost': data[data_keys[0]]['gencost'].x.size(1),
        'load': data[data_keys[0]]['load'].x.size(1),
        'shunt': data[data_keys[0]]['shunt'].x.size(1)
    }

    if args.backend == 'hgt':
        model = HGT(
            metadata=metadata,
            input_channels=input_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers
        ).to(device)
    else:
        model = HeteroGNN(
            metadata=metadata,
            input_channels=input_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            backend=args.backend,
        ).to(device)

    matpower_data = process_matpower_file("./data/pglib/pglib_opf_case14_ieee.m")
    sl_model = HeteroGNNLightning(model=model, lr=args.lr, matpower_data=matpower_data)

    dataset = OPFDataset(root=env_config['root'], case_name="pglib_opf_case14_ieee", num_groups=1)
    n_samples = len(dataset)
    n_train, n_val = int(n_samples * 0.8), int(n_samples * 0.1)
    n_test = n_samples - n_train - n_val
    train_ds, val_ds, test_ds = dataset[:n_train], dataset[n_train: n_train + n_val], dataset[n_train + n_val:]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename='best-checkpoint'
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback]
    )

    trainer.fit(sl_model, train_dataloaders=train_loader)
