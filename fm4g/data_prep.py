import torch
from torch_geometric.data import HeteroData
import numpy as np
from fm4g.opf import process_matpower_file, OPFDataset


def load_opf_data(case_name):
    """Load OPF data using fm4g/opf.py interface

    Feature dimensions:
    - bus: [vmin, vmax, vm, va]
    - generator: [pmin, pmax, qmin, qmax, vg, mbase, status, pgen, qgen, pc1, pc2]
    - load: [p, q]
    - shunt: [p, q]
    """
    dataset = OPFDataset(root="../pglib-opf", case_name=case_name)
    data = next(iter(dataset))  # Get the first data point
    return data


def load_matpower_data(case_name):
    """Load MATPOWER data using fm4g/opf.py interface

    Feature dimensions match OPF format through process_matpower_file function
    """
    file_path = "../pglib-opf/"
    data = process_matpower_file(file_path + case_name + ".m")

    # Extract base MVA for scaling if needed
    # base_mva = data.baseMVA

    return data
