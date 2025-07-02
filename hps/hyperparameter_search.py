"""
Hyperparameter Search Script using DeepHyper for OPF HeteroGNN Model

This script uses DeepHyper to perform automated hyperparameter optimization
for the HeteroGNN model on OPF datasets.

NOTE: this script applies the Ray backend for distributed hyperparameter optimization, additional packages needed:
- deephyper
- ray
- mpi4py
- ipywidgets
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DeepHyper imports
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

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


def train_and_evaluate_model(config, case_name, device, dataset, metadata, input_channels,
                             max_epochs=50, patience=5):
    """Train and evaluate model with given hyperparameters"""

    # Create data splits
    n_samples = len(dataset)
    n_train = int(n_samples * 0.8)  # Fixed train split
    n_val = int(n_samples * 0.1)   # Fixed val split

    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
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
    model = HeteroGNN(
        metadata=metadata,
        input_channels=input_channels,
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        backend=config['backend']
    )

    # Initialize model
    sample_data = next(iter(dataset))
    model = initialize_model(model, sample_data, device)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2']),
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            bus_loss = F.mse_loss(out['bus'], batch['bus'].y)
            gen_loss = F.mse_loss(out['generator'], batch['generator'].y)
            loss = bus_loss + gen_loss
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_samples += 1

        avg_train_loss = train_loss_total / train_samples

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict)
                bus_loss = F.mse_loss(out['bus'], batch['bus'].y)
                gen_loss = F.mse_loss(out['generator'], batch['generator'].y)
                loss = bus_loss + gen_loss

                val_loss_total += loss.item()
                val_samples += 1

        avg_val_loss = val_loss_total / val_samples

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Test evaluation
    model.eval()
    test_loss_total = 0.0
    test_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            bus_loss = F.mse_loss(out['bus'], batch['bus'].y)
            gen_loss = F.mse_loss(out['generator'], batch['generator'].y)
            loss = bus_loss + gen_loss

            test_loss_total += loss.item()
            test_samples += 1

    avg_test_loss = test_loss_total / test_samples

    return {
        'val_loss': best_val_loss,
        'test_loss': avg_test_loss,
        'epochs_trained': epoch
    }


def objective_function(config):
    """
    Objective function for DeepHyper optimization.
    Returns the validation loss to minimize.
    """
    try:
        # Get case name from environment or use default
        case_name = os.environ.get('CASE_NAME', 'pglib_opf_case14_ieee')

        print(f"Evaluating config: {config}")
        print(f"Training on case: {case_name}")

        # Load base config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        dataset = OPFDataset(
            root=base_config['root'],
            case_name=case_name,
            num_groups=20,
            force_reload=False
        )
        metadata = dataset.metadata()
        sample_data = next(iter(dataset))

        input_channels = {
            'bus': sample_data['bus'].x.size(1),
            'generator': sample_data['generator'].x.size(1),
            'load': sample_data['load'].x.size(1),
            'shunt': sample_data['shunt'].x.size(1)
        }

        # Train and evaluate model
        results = train_and_evaluate_model(
            config=config,
            case_name=case_name,
            device=device,
            dataset=dataset,
            metadata=metadata,
            input_channels=input_channels,
            max_epochs=100,  # Increased for more thorough training
            patience=10
        )

        print(f"Results: {results}")

        # Return negative validation loss (since DeepHyper maximizes)
        return -results['val_loss']

    except Exception as e:
        print(f"Error in objective function: {e}")
        import traceback
        traceback.print_exc()
        return -float('inf')  # Return worst possible score on error


def create_hyperparameter_problem():
    """Create a smaller hyperparameter search space for quick testing"""
    problem = HpProblem()

    # Model hyperparameters
    problem.add_hyperparameter(["sage", "gin", "gcn", "gat"], "backend", default_value="sage")
    problem.add_hyperparameter([16, 32, 64, 128, 256], "hidden_channels", default_value=64)
    problem.add_hyperparameter([2, 3, 4, 5, 6, 7, 8], "num_layers", default_value=4)

    # Optimizer hyperparameters
    problem.add_hyperparameter((1e-4, 1e-3, "log-uniform"), "lr", default_value=1e-4)
    problem.add_hyperparameter((0.8, 0.99), "beta1", default_value=0.9)
    problem.add_hyperparameter((0.9, 0.999), "beta2", default_value=0.999)
    problem.add_hyperparameter((1e-10, 1e-6, "log-uniform"), "eps", default_value=1e-8)
    problem.add_hyperparameter((0.0, 1e-3), "weight_decay", default_value=0.0)

    # Training hyperparameters
    problem.add_hyperparameter([16, 32, 64, 128], "batch_size", default_value=32)

    return problem


def run_hyperparameter_search(case_name, max_evals=100, random_state=42):
    """Run the hyperparameter search"""

    # Set environment variable for case name
    os.environ['CASE_NAME'] = case_name

    # Create problem
    problem = create_hyperparameter_problem()

    # Create results directory
    results_dir = Path(f"results_{case_name}")
    results_dir.mkdir(exist_ok=True)

    print(f"Starting hyperparameter search for {case_name}")
    print(f"Search space: {len(problem)} hyperparameters")
    print(f"Maximum evaluations: {max_evals}")
    print(f"Results will be saved to: {results_dir}")

    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }
    is_gpu_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()

    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1  # Adjust based on your GPU memory

    # Create evaluator
    evaluator = Evaluator.create(
        objective_function,
        method="ray",
        method_kwargs=method_kwargs
    )

    # Create and run search
    # TODO: replace with DistributedCBO if using distributed setup
    search = CBO(
        problem,
        evaluator,
        random_state=random_state,
        log_dir=str(results_dir)
    )
    results = search.search(max_evals=max_evals)

    # Save results
    results_file = results_dir / "search_results.csv"
    results.to_csv(results_file, index=False)

    # Find and save best configuration
    best_idx = results["objective"].idxmax()
    best_config = results.iloc[best_idx]

    best_config_dict = {
        'backend': best_config['p:backend'],
        'hidden_channels': int(best_config['p:hidden_channels']),
        'num_layers': int(best_config['p:num_layers']),
        'lr': best_config['p:lr'],
        'beta1': best_config['p:beta1'],
        'beta2': best_config['p:beta2'],
        'eps': best_config['p:eps'],
        'weight_decay': best_config['p:weight_decay'],
        'batch_size': int(best_config['p:batch_size']),
        'objective_value': best_config['objective'],
        'case_name': case_name
    }

    best_config_file = results_dir / "best_config.json"
    with open(best_config_file, 'w') as f:
        json.dump(best_config_dict, f, indent=2)

    print(f"\nHyperparameter search completed!")
    print(f"Best objective value: {best_config['objective']:.6f}")
    print(f"Best configuration saved to: {best_config_file}")
    print(f"Full results saved to: {results_file}")

    return results, best_config_dict


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for OPF HeteroGNN')
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
    parser.add_argument('--max_evals', type=int, default=100,
                        help='Maximum number of evaluations')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')

    args = parser.parse_args()

    # Check if DeepHyper is installed
    try:
        import deephyper
        print(f"DeepHyper version: {deephyper.__version__}")
    except ImportError:
        print("DeepHyper is not installed. Please install it with:")
        print("pip install deephyper")
        sys.exit(1)

    # Run hyperparameter search
    results, best_config = run_hyperparameter_search(
        case_name=args.case_name,
        max_evals=args.max_evals,
        random_state=args.random_state
    )

    print("\nBest configuration found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
