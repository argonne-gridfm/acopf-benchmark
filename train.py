import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from fm4g.constraints import (compute_generation_cost,
                              compute_line_limit_violation_v2,
                              compute_power_flow_violation_v2)
from fm4g.models.hgnn import HGT, HeteroGNN
from fm4g.opf import OPFDataset, process_matpower_file
from fm4g.opf_syn import OPFSynDataset


class HeteroGNNLightning(pl.LightningModule):
    def __init__(self, model, lr=0.001, matpower_data=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.matpower_data = matpower_data
        # NOTE: use registrer_buffer to enable automatically move to device
        self.Y_real = torch.tensor(matpower_data.Y.real.todense(), dtype=torch.float32)
        self.Y_imag = torch.tensor(matpower_data.Y.imag.todense(), dtype=torch.float32)
        self.load_bus_indices = matpower_data.load_bus_indices

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

        # NOTE: compute the violations
        # Convert to per unit
        base_mva = batch['baseMVA'].item()
        pg_pred_pu = pg_pred / base_mva
        qg_pred_pu = qg_pred / base_mva
        pd = batch['load'].x[:, 0] / base_mva
        qd = batch['load'].x[:, 1] / base_mva

        # Power injections
        P_inj = -pd
        Q_inj = -qd
        P_inj[batch['bus', 'generator_link', 'generator'].edge_index[0, :]] += pg_pred_pu
        Q_inj[batch['bus', 'generator_link', 'generator'].edge_index[0, :]] += qg_pred_pu

        rate_a = batch['bus', 'ac_line', 'bus'].edge_attr[:, 6]
        gencost_coeff = batch['generator'].x[:, -3:]
        edge_index = batch['ac_line'].edge_index

        bus_loss = compute_power_flow_violation_v2(
            vm_pred,
            va_pred,
            P_inj,
            Q_inj,
            self.Y_real,
            self.Y_imag,
            self.load_bus_indices)
        gen_loss = compute_generation_cost(gencost_coeff, pg_pred)
        line_loss = compute_line_limit_violation_v2(vm_pred, va_pred, self.Y_real, self.Y_imag, rate_a, edge_index)

        loss = bus_mse + gen_mse
        self.log('val_loss', loss)
        self.log('phy_loss', bus_loss + line_loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class OPFModel(pl.LightningModule):
    r''' OFPModel for ACOPF task, it is a wrapper of HGNN models. '''

    def __init__(self, metadata, input_channels, hidden_channels=128, num_layers=12, lr=1e-3):
        super().__init__()
        self.model = HGT(
            metadata,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers)
        self.lr = lr
        self.save_hyperparameters(ignore=['metadata'])

    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)

    def _compute_losses(self, batch, outputs):
        # ACOPF parameters
        base_mva = batch.baseMVA.unique().item()
        vmax = batch['bus'].x[:, 2]
        vmin = batch['bus'].x[:, 3]
        qmax = batch['generator'].x[:, 0]
        qmin = batch['generator'].x[:, 1]
        pmax = batch['generator'].x[:, 4]
        pmin = batch['generator'].x[:, 5]
        gencost = batch['gencost'].x
        rate_a = batch['bus', 'ac_line', 'bus'].edge_attr[:, 3]

        # Convert predicted voltage to actual values
        vm_pred = outputs['bus'][:, 0] * (vmax - vmin) + vmin
        va_pred = outputs['bus'][:, 1] * 360 - 180

        # Calculate predicted active and reactive power for generators
        pg_pred = outputs['generator'][:, 0] * (pmax - pmin) + pmin
        qg_pred = outputs['generator'][:, 1] * (qmax - qmin) + qmin

        # Convert predicted generator power to per unit (p.u.)
        pg_pred_pu = pg_pred / base_mva
        qg_pred_pu = qg_pred / base_mva
        pd_pu = batch['load'].x[:, 0] / base_mva
        qd_pu = batch['load'].x[:, 1] / base_mva

        # Initialize power injection vectors
        P_inj = -pd_pu
        Q_inj = -qd_pu

        # power flow
        loss_bus = compute_power_flow_violation_v2(vm_pred,
                                                   va_pred,
                                                   P_inj,
                                                   Q_inj,
                                                   batch.Y_real,
                                                   batch.Y_imag,
                                                   normalize=True)
        loss_gen = compute_generation_cost(gencost,
                                           pg_pred,
                                           normalize=True)
        loss_line = compute_line_limit_violation_v2(vm_pred,
                                                    va_pred,
                                                    batch.Y_real,
                                                    batch.Y_imag,
                                                    rate_a,
                                                    batch['bus',
                                                          'ac_line',
                                                          'bus'].edge_index,
                                                    normalize=True)

        total_loss = loss_bus + loss_gen + loss_line
        return total_loss, loss_bus, loss_gen, loss_line

    def training_step(self, batch, batch_idx):
        outputs = self(batch.x_dict, batch.edge_index_dict)
        total_loss, loss_bus, loss_gen, loss_line = self._compute_losses(batch, outputs)

        # Log losses - manually specify batch_size=1 to avoid automatic detection
        self.log('train_loss', total_loss, batch_size=1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_bus', loss_bus, batch_size=1, on_step=True, on_epoch=True)
        self.log('train_loss_gen', loss_gen, batch_size=1, on_step=True, on_epoch=True)
        self.log('train_loss_line', loss_line, batch_size=1, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch.x_dict, batch.edge_index_dict)
        total_loss, loss_bus, loss_gen, loss_line = self._compute_losses(batch, outputs)

        # Log losses - manually specify batch_size=1
        self.log('val_loss', total_loss, batch_size=1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_bus', loss_bus, batch_size=1, on_step=False, on_epoch=True)
        self.log('val_loss_gen', loss_gen, batch_size=1, on_step=False, on_epoch=True)
        self.log('val_loss_line', loss_line, batch_size=1, on_step=False, on_epoch=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch.x_dict, batch.edge_index_dict)
        total_loss, loss_bus, loss_gen, loss_line = self._compute_losses(batch, outputs)

        # Log losses - manually specify batch_size=1
        self.log('test_loss', total_loss, batch_size=1, on_step=False, on_epoch=True)
        self.log('test_loss_bus', loss_bus, batch_size=1, on_step=False, on_epoch=True)
        self.log('test_loss_gen', loss_gen, batch_size=1, on_step=False, on_epoch=True)
        self.log('test_loss_line', loss_line, batch_size=1, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def configure_callbacks(self):
        # TODO: add callbacks for early stopping, model checkpointing, etc.
        return super().configure_callbacks()


class OPFDataModule(pl.LightningDataModule):
    r""" OPFDataModule for loading and splitting the OPF dataset, a way of decoupling data-related hooks. """

    def __init__(self, root="./data", case_name="pglib_opf_case14_ieee", batch_size=128):
        super().__init__()
        self.root = root
        self.case_name = case_name
        self.batch_size = batch_size
        self.dataset = None

    def prepare_data(self):
        # NOTE: no need to download the dataset
        pass
        # OPFSynDataset(root=self.root, case_name=self.case_name)

    def setup(self, stage=None):
        r""" Prepare the dataset and splits. """
        # Load the dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = OPFSynDataset(root=self.root, case_name=self.case_name).to(device)

        # Split the dataset
        n_samples = len(self.dataset)
        # TODO: replace with train_split/val_split/test_split in config.yaml
        n_train, n_val = int(n_samples * 0.8), int(n_samples * 0.1)
        # n_test = n_samples - n_train - n_val
        self.train_ds = self.dataset[:n_train]
        self.val_ds = self.dataset[n_train:n_train + n_val]
        self.test_ds = self.dataset[n_train + n_val:]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid PyG DataLoader warnings
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    @property
    def metadata(self):
        return self.dataset.metadata()


def run_opf_task():
    """Run the OPF task from demo_lightning_train.py"""
    with open("config.yaml", "r") as config_file:
        env_config = yaml.safe_load(config_file)

    # Load dataset
    ds1 = OPFDataset(root=env_config["root"], case_name="pglib_opf_case14_ieee")
    metadata = ds1.metadata()
    data = next(iter(ds1))

    # Model configuration
    hidden_channels = 64
    num_layers = 3

    input_channels = {
        'bus': data['bus'].x.size(1),
        'generator': data['generator'].x.size(1),
        'load': data['load'].x.size(1),
        'shunt': data['shunt'].x.size(1)
    }

    # Create model
    model = HeteroGNN(metadata=metadata,
                      input_channels=input_channels,
                      hidden_channels=hidden_channels,
                      num_layers=num_layers)

    # Process matpower data and create Lightning module
    matpower_data = process_matpower_file("./data/pglib/pglib_opf_case14_ieee.m")
    hgnn_model = HeteroGNNLightning(model=model, lr=0.001, matpower_data=matpower_data)

    # Data loaders
    train_loader = DataLoader(ds1, batch_size=512, shuffle=True)
    val_loader = DataLoader(ds1, batch_size=1, shuffle=False)
    test_loader = DataLoader(ds1, batch_size=1, shuffle=False)

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename='opf-best-checkpoint'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback]
    )

    # Train
    trainer.fit(hgnn_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.fit(hgnn_model, datamodule=)
    # trainer.validate(hgnn_model, dataloaders=val_loader)

    trainer.test(hgnn_model, dataloaders=test_loader)
    # trainer.predict(hgnn_model, dataloaders=val_loader)


def run_opf_syn_task():
    """Run the OPF-Syn task from demo_opfsyn.py"""

    with open("config.yaml", "r") as config_file:
        env_config = yaml.safe_load(config_file)

    # Data module
    data_module = OPFDataModule(root="./data", case_name="pglib_opf_case14_ieee", batch_size=128)
    data_module.prepare_data()
    data_module.setup()

    # Model setup
    input_channels = {
        'bus': 4,
        'generator': 6,
        'load': 2,
        'shunt': 2,
    }

    model = OPFModel(
        metadata=data_module.metadata,
        input_channels=input_channels,
        hidden_channels=64,
        num_layers=3,
        lr=1e-3
    )

    # Logger
    logger = TensorBoardLogger(save_dir="lightning_logs", name="opf_syn")

    # Callbacks - save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="opfsyn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # Automatically select the accelerator (CPU or GPU)
        devices="auto",  # if not defined, it will use all available GPUs
        num_nodes=1,  # num of gpus nodes for distributed training
        logger=logger,
        accumulate_grad_batches=4,
        # Gradient accumulation steps (effective batch size = batch_size * accumulate_grad_batches)
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        fast_dev_run=True,
        gradient_clip_val=None,  # Gradient clipping value if needed
        precision=None,
        profiler=None,
        strategy="ddp",
        inference_mode=True,
    )

    # Training
    trainer.fit(model, data_module)

    # Testing
    test_results = trainer.test(model, data_module)
    print(f"Test results: {test_results}")


def main():
    parser = argparse.ArgumentParser(description='Run supervised or unsupservised tasks')
    parser.add_argument('--task', type=str, choices=['supervised', 'unsupervised'], default='unsupervised',
                        help='Task to run: supervised (with optimal labels) or unsupervised (w/o labels)')
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    args = parser.parse_args()

    # for reproducibility
    if args.seed > 0:
        seed_everything(args.seed, workers=True)

    if args.task == 'supervised':
        print("Running OPF task...")
        run_opf_task()
        # raise NotImplementedError("Supervised task is under debug")
        # pass
    elif args.task == 'unsupervised':
        print("Running OPF-Syn task...")
        run_opf_syn_task(args)


if __name__ == "__main__":
    main()
