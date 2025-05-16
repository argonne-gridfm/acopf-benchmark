import argparse
import random
import csv
import os
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import yaml
from tqdm import tqdm

from fm4g.constraints import (compute_generation_cost,
                              compute_line_limit_violation,
                              compute_power_flow_violation)
from fm4g.models.hgnn import HGT, HeteroGNN
from fm4g.opf import process_matpower_file


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data, load, batch_size):
        self.data = data
        self.load = load
        self.batch_size = batch_size
        self.keys = list(data.keys())

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        case_name = random.choice(self.keys)
        perturbation = (torch.rand_like(self.load[case_name]) - 0.5) * 0.2
        pd = self.load[case_name][:, 0] * (1.0 + perturbation[:, 0])
        qd = self.load[case_name][:, 1] * (1.0 + perturbation[:, 1])
        self.data[case_name]['load'].x = torch.stack([pd, qd], dim=1)
        return self.data[case_name]


class GridFMLightning(pl.LightningModule):
    def __init__(self, model, args, data, load, Y_real, Y_imag, base_mva,
                 vmax, vmin, qmax, qmin, pmax, pmin, gencost, rate_a):
        super().__init__()
        self.model = model
        self.args = args
        self.data = data
        self.load = load
        self.Y_real = Y_real
        self.Y_imag = Y_imag
        self.base_mva = base_mva
        self.vmax = vmax
        self.vmin = vmin
        self.qmax = qmax
        self.qmin = qmin
        self.pmax = pmax
        self.pmin = pmin
        self.gencost = gencost
        self.rate_a = rate_a
        self.training_metrics = []
        self.instance_sample_count = {key: 0 for key in self.data.keys()}

    def train_dataloader(self):
        dataset = RandomDataset(self.data, self.load, self.args.batch_size)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=lambda x: x[0]
        )

    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)

    def training_step(self, batch, batch_idx):
        case_name = next(key for key, value in self.data.items() if value == batch)
        self.instance_sample_count[case_name] += 1

        out = self(batch.x_dict, batch.edge_index_dict)

        vm_pred = out['bus'][:, 0] * (self.vmax[case_name] - self.vmin[case_name]) + self.vmin[case_name]
        va_pred = out['bus'][:, 1] * 360 - 180
        pg_pred = out['generator'][:, 0] * (self.pmax[case_name] - self.pmin[case_name]) + self.pmin[case_name]
        qg_pred = out['generator'][:, 1] * (self.qmax[case_name] - self.qmin[case_name]) + self.qmin[case_name]

        pd = batch['load'].x[:, 0]
        qd = batch['load'].x[:, 1]
        P_inj = -pd / self.base_mva[case_name]
        Q_inj = -qd / self.base_mva[case_name]
        gen_indices = self.data[case_name]['bus', 'generator_link', 'generator'].edge_index[0, :]
        P_inj[gen_indices] += pg_pred / self.base_mva[case_name]
        Q_inj[gen_indices] += qg_pred / self.base_mva[case_name]

        bus_loss = compute_power_flow_violation(
            vm_pred, va_pred, P_inj, Q_inj, self.Y_real[case_name], self.Y_imag[case_name])
        gen_loss = compute_generation_cost(self.gencost[case_name], pg_pred)
        line_loss = compute_line_limit_violation(
            vm_pred, va_pred, self.Y_real[case_name], self.Y_imag[case_name], self.rate_a[case_name],
            self.data[case_name]['bus', 'ac_line', 'bus'].edge_index
        )

        loss = bus_loss + self.args.gamma_1 * line_loss + self.args.gamma_2 * gen_loss

        self.log('train_loss', loss)
        self.log('bus_loss', bus_loss)
        self.log('line_loss', line_loss)
        self.log('gen_loss', gen_loss)

        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss'].item()
        avg_bus_loss = self.trainer.callback_metrics['bus_loss'].item()
        avg_line_loss = self.trainer.callback_metrics['line_loss'].item()
        avg_gen_loss = self.trainer.callback_metrics['gen_loss'].item()

        self.training_metrics.append([
            self.current_epoch, avg_loss, avg_bus_loss, avg_line_loss, avg_gen_loss
        ])

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)


def parse_args():
    r""" Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='GridFM Training Script')
    parser.add_argument('--backend', type=str, default='sage', choices=['sage', 'gcn', 'gat', 'gin', 'hgt'],
                        help='Backend GNN architecture')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gamma_1', type=float, default=1.0,
                        help='Weight for line loss')
    parser.add_argument('--gamma_2', type=float, default=1e-10,
                        help='Weight for generation loss')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--test_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def set_seed(seed):
    r"""Set random seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    with open("config.yaml", "r") as config_file:
        env_config = yaml.safe_load(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = "./data/pglib/"
    matpower_instances = {
        "case14_ieee": "pglib_opf_case14_ieee.m",
    }

    # Process all instances
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
    train_data_keys = test_data_keys = data_keys[:1]

    # Initialize model
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

    model(data[data_keys[0]].x_dict, data[data_keys[0]].edge_index_dict)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

    # Create Lightning module
    lightning_model = GridFMLightning(
        model=model,
        args=args,
        data=data,
        load=load,
        Y_real=Y_real,
        Y_imag=Y_imag,
        base_mva=base_mva,
        vmax=vmax,
        vmin=vmin,
        qmax=qmax,
        qmin=qmin,
        pmax=pmax,
        pmin=pmin,
        gencost=gencost,
        rate_a=rate_a
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    # Train model
    trainer.fit(lightning_model)

    # Save training metrics
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    csv_filename = f"{args.backend}_{args.hidden_channels}_{args.num_layers}_{args.lr}.csv"
    csv_path = log_dir / csv_filename

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'total_loss', 'bus_loss', 'line_loss', 'gen_loss'])
        writer.writerows(lightning_model.training_metrics)

    # Print training samples per instance
    print("\nTraining samples per instance:")
    for case_name, count in lightning_model.instance_sample_count.items():
        print(f'Instance {case_name}: {count}')
