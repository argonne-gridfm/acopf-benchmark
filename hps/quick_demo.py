"""
Quick hyperparameter search with a smaller search space for testing for debugging purposes.
This script allows for rapid testing of the hyperparameter search setup without the full complexity of the original
hyperparameter search script.
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deephyper.evaluator import Evaluator
from deephyper.hpo import HpProblem
from hps.hyperparameter_search import (create_hyperparameter_problem,
                                       run_hyperparameter_search)


def create_quick_search_problem():
    """Create a smaller hyperparameter search space for quick testing"""
    problem = HpProblem()

    # Reduced search space for quick testing
    problem.add_hyperparameter(["sage", "gin"], "backend", default_value="sage")
    problem.add_hyperparameter([32, 64, 128], "hidden_channels", default_value=64)
    problem.add_hyperparameter([3, 4, 5], "num_layers", default_value=4)

    # Optimizer hyperparameters (reduced ranges)
    problem.add_hyperparameter((1e-4, 1e-3, "log-uniform"), "lr", default_value=1e-4)
    problem.add_hyperparameter((0.85, 0.95), "beta1", default_value=0.9)
    problem.add_hyperparameter((0.95, 0.999), "beta2", default_value=0.999)
    problem.add_hyperparameter((1e-8, 1e-7, "log-uniform"), "eps", default_value=1e-8)
    problem.add_hyperparameter((0.0, 1e-4), "weight_decay", default_value=0.0)

    # Fixed batch size for simplicity
    problem.add_hyperparameter([32, 64], "batch_size", default_value=32)

    return problem


def main():
    parser = argparse.ArgumentParser(description='Quick Hyperparameter Search for OPF HeteroGNN')
    parser.add_argument('--case_name', type=str, default='pglib_opf_case14_ieee',
                        help='Case name for the dataset')
    parser.add_argument('--max_evals', type=int, default=20,
                        help='Maximum number of evaluations (default: 20 for quick search)')

    args = parser.parse_args()

    print("Running quick hyperparameter search...")
    print(f"Case: {args.case_name}")
    print(f"Max evaluations: {args.max_evals}")

    # Monkey patch the problem creation function for quick search
    # import hyperparameter_search
    # hyperparameter_search.create_hyperparameter_problem = create_quick_search_problem

    # Run search
    results, best_config = run_hyperparameter_search(
        case_name=args.case_name,
        max_evals=args.max_evals,
        random_state=42
    )

    print("\nQuick search completed!")
    print("Best configuration found:")
    for key, value in best_config.items():
        if key != 'case_name':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
