import argparse
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from fm4g.models.hgnn import HeteroGNN
from fm4g.opf import OPFDataset


def initialize_model(model, sample_data, device):
    """Lazy initialize model parameters by running a dummy forward pass"""
    print("Initializing model parameters...")
    model = model.to(device)
    sample_data = sample_data.to(device)

    model.eval()
    with torch.no_grad():
        try:
            _ = model(sample_data.x_dict, sample_data.edge_index_dict)
            print("Model parameters initialized successfully!")
        except Exception as e:
            print(f"Warning: Model initialization failed: {e}")
            print("Model may still work during training...")

    return model


def train_model(model, train_loader, val_loader, optimizer, device, writer, epochs=100, patience=5, log_dir=None):
    """Train model with early stopping based on validation loss"""
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_bus_loss_total = 0.0
        train_gen_loss_total = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            bus_loss = F.mse_loss(out['bus'], batch['bus'].y)
            gen_loss = F.mse_loss(out['generator'], batch['generator'].y)
            loss = bus_loss + gen_loss
            loss.backward()
            optimizer.step()

            train_bus_loss_total += bus_loss.item()
            train_gen_loss_total += gen_loss.item()
            train_samples += 1

            pbar.set_postfix({'bus_loss': bus_loss.item(),
                             'gen_loss': gen_loss.item()})

        # Calculate average training losses
        avg_train_bus_loss = train_bus_loss_total / train_samples
        avg_train_gen_loss = train_gen_loss_total / train_samples
        avg_train_total_loss = avg_train_bus_loss + avg_train_gen_loss

        # Validation phase
        val_bus_loss, val_gen_loss, val_total_loss = validate_model(model, val_loader, device)

        # Log to tensorboard
        writer.add_scalar('Loss/Train_Bus', avg_train_bus_loss, epoch)
        writer.add_scalar('Loss/Train_Gen', avg_train_gen_loss, epoch)
        writer.add_scalar('Loss/Train_Total', avg_train_total_loss, epoch)
        writer.add_scalar('Loss/Val_Bus', val_bus_loss, epoch)
        writer.add_scalar('Loss/Val_Gen', val_gen_loss, epoch)
        writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)

        print(f"Epoch {epoch}: Train Loss = {avg_train_total_loss:.6f}, Val Loss = {val_total_loss:.6f}")

        # Early stopping check
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(f"{log_dir}/best_model.pth"))
    return model


def validate_model(model, val_loader, device):
    """Validate model and return average losses"""
    model.eval()
    total_bus_loss = 0.0
    total_gen_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            bus_loss = F.mse_loss(out['bus'], batch['bus'].y)
            gen_loss = F.mse_loss(out['generator'], batch['generator'].y)

            total_bus_loss += bus_loss.item()
            total_gen_loss += gen_loss.item()
            num_samples += 1

    avg_bus_loss = total_bus_loss / num_samples
    avg_gen_loss = total_gen_loss / num_samples
    avg_total_loss = avg_bus_loss + avg_gen_loss

    return avg_bus_loss, avg_gen_loss, avg_total_loss


def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset and print losses"""
    print("Evaluating model on test dataset...")
    model.eval()
    total_bus_loss = 0.0
    total_gen_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            bus_loss = F.mse_loss(out['bus'], batch['bus'].y)
            gen_loss = F.mse_loss(out['generator'], batch['generator'].y)

            total_bus_loss += bus_loss.item()
            total_gen_loss += gen_loss.item()
            num_samples += 1

    avg_bus_loss = total_bus_loss / num_samples
    avg_gen_loss = total_gen_loss / num_samples
    avg_total_loss = avg_bus_loss + avg_gen_loss

    print(f"Test Results:")
    print(f"  Bus Loss: {avg_bus_loss:.6f}")
    print(f"  Generator Loss: {avg_gen_loss:.6f}")
    print(f"  Total Loss: {avg_total_loss:.6f}")

    return avg_bus_loss, avg_gen_loss, avg_total_loss


def main():
    parser = argparse.ArgumentParser(description='Train HeteroGNN for OPF')
    parser.add_argument('--case_name', type=str, default='pglib_opf_case14_ieee',
                        choices=['pglib_opf_case14_ieee',
                                 'pglib_opf_case30_ieee',
                                 'pglib_opf_case57_ieee',
                                 'pglib_opf_case118_ieee',
                                 'pglib_opf_case500_goc',
                                 'pglib_opf_case2000_goc',
                                 'pglib_opf_case4661_sdet',
                                 'pglib_opf_case6470_rte',
                                 'pglib_opf_case10000_goc',
                                 'pglib_opf_case13659_pegase'],
                        help='Case name for the dataset')
    args = parser.parse_args()

    print(f"Training on case: {args.case_name}")

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = OPFDataset(root=config['root'],
                         case_name=args.case_name,
                         num_groups=20,
                         force_reload=False)
    metadata = dataset.metadata()
    sample_data = next(iter(dataset))

    input_channels = {
        'bus': sample_data['bus'].x.size(1),
        'generator': sample_data['generator'].x.size(1),
        'load': sample_data['load'].x.size(1),
        'shunt': sample_data['shunt'].x.size(1)
    }

    # Create data splits
    n_samples = len(dataset)
    n_train = int(n_samples * config['train_split'])
    n_val = int(n_samples * config['val_split'])

    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]

    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['loader']['batch_size'],
        shuffle=config['loader']['shuffle'],
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model_config = config['models']['HeteroGNN'].copy()
    model_config['backend'] = 'sage'

    model = HeteroGNN(
        metadata=metadata,
        input_channels=input_channels,
        hidden_channels=model_config['hidden_channels'],
        num_layers=model_config['num_layers'],
        backend=model_config['backend']
    )

    # Initialize model
    model = initialize_model(model, sample_data, device)

    # Setup optimizer
    opt_config = config['optimizer']
    opt_config['lr'] = 1.0e-04
    optimizer = torch.optim.Adam(model.parameters(), **opt_config)

    # Setup tensorboard
    log_dir = f"runs/{args.case_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    try:
        # Train model
        print("Starting training...")
        model = train_model(model, train_loader, val_loader, optimizer, device, writer,
                            epochs=100, patience=5, log_dir=log_dir)

        # Evaluate on test set
        evaluate_model(model, test_loader, device)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
