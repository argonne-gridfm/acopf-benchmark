import argparse
import random
import csv
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm

from fm4g.constraints import (compute_generation_cost,
                              compute_line_limit_violation,
                              compute_power_flow_violation)
from fm4g.models.hgnn import HGT, HeteroGNN
from fm4g.opf import process_matpower_file
from utils import set_seed, parse_args


def train_model(model, loader, args, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    instance_sample_count = {key: 0 for key in train_data_keys}

    # Initialize lists to store metrics for all epochs
    training_metrics = []

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        total_loss = total_loss_bus = total_loss_line = total_loss_gen = 0

        for _ in range(args.batch_size):
            case_name = random.choice(train_data_keys)
            instance_sample_count[case_name] += 1

            perturbation = (torch.rand_like(load[case_name]) - 0.5) * 0.2
            pd = load[case_name][:, 0] * (1.0 + perturbation[:, 0])
            qd = load[case_name][:, 1] * (1.0 + perturbation[:, 1])
            data[case_name]['load'].x = torch.stack([pd, qd], dim=1)

            out = model(data[case_name].x_dict, data[case_name].edge_index_dict)

            vm_pred = out['bus'][:, 0] * (vmax[case_name] - vmin[case_name]) + vmin[case_name]
            va_pred = out['bus'][:, 1] * 360 - 180
            pg_pred = out['generator'][:, 0] * (pmax[case_name] - pmin[case_name]) + pmin[case_name]
            qg_pred = out['generator'][:, 1] * (qmax[case_name] - qmin[case_name]) + qmin[case_name]

            P_inj = -pd / base_mva[case_name]
            Q_inj = -qd / base_mva[case_name]
            gen_indices = data[case_name]['bus', 'generator_link', 'generator'].edge_index[0, :]
            P_inj[gen_indices] += pg_pred / base_mva[case_name]
            Q_inj[gen_indices] += qg_pred / base_mva[case_name]

            bus_loss = compute_power_flow_violation(
                vm_pred, va_pred, P_inj, Q_inj, Y_real[case_name], Y_imag[case_name])
            gen_loss = compute_generation_cost(gencost[case_name], pg_pred)
            line_loss = compute_line_limit_violation(
                vm_pred, va_pred, Y_real[case_name], Y_imag[case_name], rate_a[case_name],
                data[case_name]['bus', 'ac_line', 'bus'].edge_index
            )

            loss = bus_loss + args.gamma_1 * line_loss + args.gamma_2 * gen_loss
            total_loss += loss.item()
            total_loss_bus += bus_loss.item()
            total_loss_line += line_loss.item()
            total_loss_gen += gen_loss.item()

            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / args.batch_size
        avg_bus_loss = total_loss_bus / args.batch_size
        avg_line_loss = total_loss_line / args.batch_size
        avg_gen_loss = total_loss_gen / args.batch_size

        # Store metrics for this epoch
        training_metrics.append([epoch, avg_loss, avg_bus_loss, avg_line_loss, avg_gen_loss])

        pbar.set_postfix(loss=avg_loss, bus_loss=avg_bus_loss, line_loss=avg_line_loss, gen_loss=avg_gen_loss)

    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Create CSV filename based on model parameters
    csv_filename = f"{args.backend}_{args.hidden_channels}_{args.num_layers}_{args.lr}.csv"
    csv_path = log_dir / csv_filename

    # Write all metrics to CSV at once
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'total_loss', 'bus_loss', 'line_loss', 'gen_loss'])
        writer.writerows(training_metrics)

    # Print training samples per instance
    print("\nTraining samples per instance:")
    for case_name, count in instance_sample_count.items():
        print(f'Instance {case_name}: {count}')


def eval_model(model, loader, args, device="cpu"):
    model.eval()
    instance_metrics = {key: {'total': 0, 'bus': 0, 'line': 0, 'gen': 0} for key in test_data_keys}

    with torch.no_grad():
        for case_name in test_data_keys:
            for _ in range(args.test_samples):
                perturbation = (torch.rand_like(load[case_name]) - 0.5) * 0.2
                pd = load[case_name][:, 0] * (1.0 + perturbation[:, 0])
                qd = load[case_name][:, 1] * (1.0 + perturbation[:, 1])
                data[case_name]['load'].x = torch.stack([pd, qd], dim=1)

                out = model(data[case_name].x_dict, data[case_name].edge_index_dict)

                vm_pred = out['bus'][:, 0] * (vmax[case_name] - vmin[case_name]) + vmin[case_name]
                va_pred = out['bus'][:, 1] * 360 - 180
                pg_pred = out['generator'][:, 0] * (pmax[case_name] - pmin[case_name]) + pmin[case_name]
                qg_pred = out['generator'][:, 1] * (qmax[case_name] - qmin[case_name]) + qmin[case_name]

                P_inj = -pd / base_mva[case_name]
                Q_inj = -qd / base_mva[case_name]
                gen_indices = data[case_name]['bus', 'generator_link', 'generator'].edge_index[0, :]
                P_inj[gen_indices] += pg_pred / base_mva[case_name]
                Q_inj[gen_indices] += qg_pred / base_mva[case_name]

                bus_loss = compute_power_flow_violation(
                    vm_pred, va_pred, P_inj, Q_inj, Y_real[case_name], Y_imag[case_name])
                gen_loss = compute_generation_cost(gencost[case_name], pg_pred)
                line_loss = compute_line_limit_violation(
                    vm_pred, va_pred, Y_real[case_name], Y_imag[case_name], rate_a[case_name],
                    data[case_name]['bus', 'ac_line', 'bus'].edge_index
                )
                eval_loss = bus_loss + args.gamma_1 * line_loss + args.gamma_2 * gen_loss
                # eval_loss = bus_loss + args.gamma * line_loss + args.gamma_2
                instance_metrics[case_name]['total'] += eval_loss.item()
                instance_metrics[case_name]['bus'] += bus_loss.item()
                instance_metrics[case_name]['line'] += line_loss.item()
                instance_metrics[case_name]['gen'] += gen_loss.item()

    total_metrics = {k: sum(inst[k] for inst in instance_metrics.values()) for k in ['total', 'bus', 'line']}
    n_total = len(test_data_keys) * args.test_samples

    print(f'\nOverall metrics:')
    print(f'Total Loss: {total_metrics["total"]/n_total:.4f}')
    print(f'Bus Loss: {total_metrics["bus"]/n_total:.4f}')
    print(f'Line Loss: {total_metrics["line"]/n_total:.4f}')

    print('\nPer instance metrics:')
    for case_name in test_data_keys:
        metrics = {k: v / args.test_samples for k, v in instance_metrics[case_name].items()}
        print(f'{case_name} - Total: {metrics["total"]:.4f}, Bus: {metrics["bus"]:.4f}, Line: {metrics["line"]:.4f}')


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    with open("config.yaml", "r") as config_file:
        env_config = yaml.safe_load(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = "./data/pglib/"
    matpower_instances = {
        "case14_ieee": "pglib_opf_case14_ieee.m",
        # "case30_ieee": "pglib_opf_case30_ieee.m",
        # "case57_ieee": "pglib_opf_case57_ieee.m",
        # "case118_ieee": "pglib_opf_case118_ieee.m",
        # "case2000_goc": "pglib_opf_case2000_goc.m",
        # "case1354_pegase": "pglib_opf_case1354_pegase.m",
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
    # train and test on the same grids
    train_data_keys = test_data_keys = data_keys[:1]
    # train on first three grids and test on the rest
    test_data_keys = data_keys[1:]
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

    train_model(model, data, args, device=device)
    eval_model(model, data, args, device=device)
