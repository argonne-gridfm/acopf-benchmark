""" Synthetic OPF dataset generator in Pytorch Geometry format.
Copyright(c) 2025, Argonne National Laboratory
All rights reserved.
"""

import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import numpy as np
import pandapower as pp
import torch
from pandapower.pypower.idx_brch import (BR_B, BR_R, BR_X, F_BUS, RATE_A,
                                         RATE_B, RATE_C, SHIFT, T_BUS, TAP)
from pandapower.pypower.idx_bus import (BASE_KV, BS, BUS_AREA, BUS_I, BUS_TYPE,
                                        GS, PD, QD, VA, VM, VMAX, VMIN, ZONE)
from pandapower.pypower.idx_cost import (COST, MODEL, NCOST, POLYNOMIAL,
                                         PW_LINEAR, SHUTDOWN, STARTUP)
from pandapower.pypower.idx_gen import (GEN_BUS, GEN_STATUS, MBASE, PG, PMAX,
                                        PMIN, QG, QMAX, QMIN, VG)
from torch_geometric.data import Dataset, HeteroData, InMemoryDataset
from tqdm import tqdm

from fm4g.constraints import convert_to_torch_sparse
from fm4g.opf import extract_edge_index, extract_edge_index_rev


class OPFSynDataset(InMemoryDataset):
    r"""Synthetic OPF dataset generator that creates perturbed versions of MATPOWER cases.
       Saves each sample individually for better scalability.

    Args:
        root (str): Root directory where the dataset should be saved.
        case_name (str): Name of the MATPOWER case (without extension).
        num_samples (int, optional): Number of synthetic samples to generate.
            (default: :obj:`15000`)
        transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.HeteroData` object and returns a transformed
            version. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.HeteroData` object and returns a transformed
            version. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        root: str,
        case_name: str,
        num_samples: int = 15000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.case_name = case_name

        cur_dir = osp.dirname(osp.abspath(__file__))
        # GLOBAL attributes
        self.matpower_file = osp.join(cur_dir, "..", "data/pglib/", self.case_name + ".m")
        if not osp.exists(self.matpower_file):
            raise FileNotFoundError(f"MATPOWER case file not found: {self.matpower_file}")
        self.net = pp.converter.from_mpc(self.matpower_file)
        self.ppc = pp.converter.pypower.to_ppc(self.net, init='flat')
        Y_sp, _, _ = pp.makeYbus_pypower(self.ppc['baseMVA'], self.ppc['bus'], self.ppc['branch'])
        self.Y_sp_pt_real = convert_to_torch_sparse(Y_sp.real)
        self.Y_sp_pt_imag = convert_to_torch_sparse(Y_sp.imag)

        # self.matpower_file = matpower_file
        self.num_samples = num_samples
        # Call Dataset constructor
        super().__init__(root, transform, pre_transform, force_reload=force_reload)

        # load data
        out = torch.load(self.processed_paths[0])
        self._data, self.slices = out

    @property
    def raw_file_names(self) -> List[str]:
        r""" A list of files in the `raw_dir` which needs to be found in order to skip the download.
        `raw_dir` is self.root + '/raw'.
        """
        return [f"{self.case_name}.m"]

    @property
    def processed_file_names(self) -> List[str]:
        r""" A list of files in the `processed_dir` which needs to be found in order to skip the processing.
        `processed_dir` is self.root + '/processed'.
        """
        return f'data_{self.case_name}.pt'

    def download(self):
        r""" Download the dataset if it is not already present.
        To ensure the raw file is ready, copy from `data/pglib` to `raw_dir`.
        """
        # os.makedirs(self.raw_dir, exist_ok=True)
        # shutil.copy(self.matpower_file, self.raw_dir)
        pass

    def process(self):
        r""" Process the raw data and save it to `processed_dir`.
        """
        # build a base case
        base_data = self.process_matpower_file()

        # Ensure processed directory exists, Dataset.__init__() creates it
        # os.makedirs(self.processed_dir, exist_ok=True)

        # Generate synthetic samples, make random perturbations to the dataset
        data_list = self._generate_sample(base_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # Save data_list to torch file, processed_paths from `Dataset`
        self._data, self.slices = self.collate(data_list)
        torch.save((self._data, self.slices), self.processed_paths[0])

    def len(self):
        # Returns the number of examples in your dataset.
        return self.num_samples

    # def get(self):
    #     data_list = torch.load(osp.join(self.processed_dir, 'data.pt'))
    #     return data_list

    def process_matpower_file(self):
        r""" Process the MATPOWER file and return the base data with HeteroData format.
        """
        obj = {
            'grid': {
                'context': [[self.ppc['baseMVA']]],
                'nodes': {
                    'bus': self.ppc['bus'][:, [BUS_TYPE, BASE_KV, VMAX, VMIN]].astype(np.float32),
                    'generator': self.ppc['gen'][:, [QMAX, QMIN, VG, MBASE, PMAX, PMIN]].astype(np.float32),
                    'gencost': np.pad(self.ppc['gencost'], ((0, 0), (0, 7 - self.ppc['gencost'].shape[1])), 'constant', constant_values=0).astype(np.float32),
                    'load': self.ppc['bus'][:, [PD, QD]].astype(np.float32),
                    'shunt': self.ppc['bus'][:, [GS, BS]].astype(np.float32)
                },
                'edges': {
                    'ac_line': {
                        'senders': self.ppc['branch'][:, F_BUS].astype(int),
                        'receivers': self.ppc['branch'][:, T_BUS].astype(int),
                        # TAP, SHIFT are at indices 8, 9
                        'features': self.ppc['branch'][:, [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT]].astype(np.float32)
                    },
                    'transformer': {
                        'senders': self.ppc['branch'][self.ppc['branch'][:, 8] != 0][:, F_BUS].astype(int),
                        'receivers': self.ppc['branch'][self.ppc['branch'][:, 8] != 0][:, T_BUS].astype(int),
                        'features': self.ppc['branch'][self.ppc['branch'][:, 8] != 0][:, [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT]].astype(np.float32)
                    },
                    'generator_link': {
                        'senders': np.arange(self.ppc['gen'].shape[0]),
                        'receivers': self.ppc['gen'][:, GEN_BUS].astype(int)
                    },
                    'load_link': {
                        'senders': self.ppc['bus'][:, BUS_I].astype(int),
                        'receivers': self.ppc['bus'][:, BUS_I].astype(int)
                    },
                    'shunt_link': {
                        'senders': self.ppc['bus'][:, BUS_I].astype(int),
                        'receivers': self.ppc['bus'][:, BUS_I].astype(int)
                    }
                }
            },
            'metadata': {
                'objective': 0.0  # not given from MATPOWER file
            }
        }
        grid = obj['grid']

        hdata = HeteroData()
        hdata.base_mva = self.ppc['baseMVA']
        hdata.Y_sp_real = self.Y_sp_pt_real
        hdata.Y_sp_imag = self.Y_sp_pt_imag
        hdata.load_bus_indices = self.net.load.bus.values.astype(np.int32)
        hdata.gen_bus_indices = self.net.gen.bus.values.astype(np.int32)

        # Nodes (only some have a target):
        # TODO: convert the bus.type to one-hot encoding features
        hdata['bus'].x = torch.tensor(grid['nodes']['bus'])
        hdata['generator'].x = torch.tensor(grid['nodes']['generator'])
        hdata['gencost'].x = torch.tensor(grid['nodes']['gencost'])
        hdata['load'].x = torch.tensor(grid['nodes']['load'])
        hdata['shunt'].x = torch.tensor(grid['nodes']['shunt'])

        # Edges (only ac lines and transformers have features):
        hdata['bus', 'ac_line', 'bus'].edge_index = extract_edge_index(obj, 'ac_line')
        hdata['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(grid['edges']['ac_line']['features'])
        # data['bus', 'ac_line', 'bus'].edge_label = torch.tensor(solution['edges']['ac_line']['features'])

        hdata['bus', 'transformer', 'bus'].edge_index = extract_edge_index(obj, 'transformer')
        hdata['bus', 'transformer', 'bus'].edge_attr = torch.tensor(grid['edges']['transformer']['features'])
        # data['bus', 'transformer', 'bus'].edge_label = torch.tensor(solution['edges']['transformer']['features'])
        # print("Warning: edge_label is not available")

        hdata['generator', 'generator_link', 'bus'].edge_index = extract_edge_index(obj, 'generator_link')
        hdata['bus', 'generator_link', 'generator'].edge_index = extract_edge_index_rev(obj, 'generator_link')

        hdata['load', 'load_link', 'bus'].edge_index = extract_edge_index(obj, 'load_link')
        hdata['bus', 'load_link', 'load'].edge_index = extract_edge_index_rev(obj, 'load_link')

        hdata['shunt', 'shunt_link', 'bus'].edge_index = extract_edge_index(obj, 'shunt_link')
        hdata['bus', 'shunt_link', 'shunt'].edge_index = extract_edge_index_rev(obj, 'shunt_link')

        return hdata

    def _generate_sample(self, base_data: HeteroData) -> HeteroData:
        r"""Generate a single perturbed sample from the base case.

        Args:
            base_data (HeteroData): The base data to be perturbed.

        Returns:
            list(HeteroData): A list of perturbed samples.
        """
        data_list = []
        for _ in range(self.num_samples):
            data = base_data.clone()

            # Note: Perturb load demands (PD, QD) by ±20% uniformly
            load_x = data['load'].x
            perturb_factor = 1 + np.random.uniform(-0.2, 0.2, size=load_x.shape)
            data['load'].x = load_x * torch.tensor(perturb_factor, dtype=torch.float)

            # TODO: Perturb generator limits (PMIN, PMAX, QMIN, QMAX) by ±10%
            # Indices: QMAX=0, QMIN=1, VG=2, MBASE=3, PMAX=4, PMIN=5
            # gen_x = data['generator'].x
            # # Perturb PMAX, PMIN, QMAX, QMIN - careful not to make min > max
            # pmax_perturb = 1 + np.random.uniform(-0.1, 0.1, size=(gen_x.shape[0], 1))
            # pmin_perturb = 1 + np.random.uniform(-0.1, 0.1, size=(gen_x.shape[0], 1))
            # qmax_perturb = 1 + np.random.uniform(-0.1, 0.1, size=(gen_x.shape[0], 1))
            # qmin_perturb = 1 + np.random.uniform(-0.1, 0.1, size=(gen_x.shape[0], 1))

            # new_pmax = gen_x[:, 4:5] * torch.tensor(pmax_perturb, dtype=torch.float)
            # new_pmin = gen_x[:, 5:6] * torch.tensor(pmin_perturb, dtype=torch.float)
            # new_qmax = gen_x[:, 0:1] * torch.tensor(qmax_perturb, dtype=torch.float)
            # new_qmin = gen_x[:, 1:2] * torch.tensor(qmin_perturb, dtype=torch.float)

            # # Ensure min <= max after perturbation
            # data['generator'].x[:, 4:5] = torch.max(new_pmin, new_pmax)  # PMAX
            # data['generator'].x[:, 5:6] = torch.min(new_pmin, new_pmax)  # PMIN
            # data['generator'].x[:, 0:1] = torch.max(new_qmin, new_qmax)  # QMAX
            # data['generator'].x[:, 1:2] = torch.min(new_qmin, new_qmax)  # QMIN

            # TODO: Perturb line parameters (R, X, B) by ±5%
            # Indices: BR_R=0, BR_X=1, BR_B=2
            # if ('bus', 'ac_line', 'bus') in data.edge_types:
            #     edge_attr = data['bus', 'ac_line', 'bus'].edge_attr
            #     if edge_attr.numel() > 0:  # Check if there are any ac_lines
            #         perturb_factor = 1 + np.random.uniform(-0.05, 0.05, size=(edge_attr.shape[0], 3))
            #         edge_attr[:, :3] = edge_attr[:, :3] * torch.tensor(perturb_factor, dtype=torch.float)
            #         data['bus', 'ac_line', 'bus'].edge_attr = edge_attr  # Assign back

            # TODO: Perturb transformer parameters (R, X, B) by ±5%
            # if ('bus', 'transformer', 'bus') in data.edge_types:
            #     edge_attr = data['bus', 'transformer', 'bus'].edge_attr
            #     if edge_attr.numel() > 0:  # Check if there are any transformers
            #         perturb_factor = 1 + np.random.uniform(-0.05, 0.05, size=(edge_attr.shape[0], 3))
            #         edge_attr[:, :3] = edge_attr[:, :3] * torch.tensor(perturb_factor, dtype=torch.float)
            #         data['bus', 'transformer', 'bus'].edge_attr = edge_attr  # Assign back

            # TODO: Randomly disconnect some lines (10% probability) - Requires careful handling
            # Disconnecting lines changes the graph structure and might make the OPF unsolvable
            # or significantly different. Consider if this is desired.
            # if ('bus', 'ac_line', 'bus') in data.edge_types:
            #     num_edges = data['bus', 'ac_line', 'bus'].edge_index.shape[1]
            #     if num_edges > 0:
            #         mask = torch.rand(num_edges) > 0.1 # Keep 90%
            #         data['bus', 'ac_line', 'bus'].edge_index = data['bus', 'ac_line', 'bus'].edge_index[:, mask]
            #         data['bus', 'ac_line', 'bus'].edge_attr = data['bus', 'ac_line', 'bus'].edge_attr[mask]
            # Similar logic for transformers if desired.

            data_list.append(data)

        return data_list

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_samples={self.num_samples})')

    def metadata(self):
        return (
            ['bus', 'generator', 'load', 'shunt'],
            [
                ('bus', 'ac_line', 'bus'),
                ('bus', 'transformer', 'bus'),
                ('generator', 'generator_link', 'bus'),
                ('bus', 'generator_link', 'generator'),
                ('load', 'load_link', 'bus'),
                ('bus', 'load_link', 'load'),
                ('shunt', 'shunt_link', 'bus'),
                ('bus', 'shunt_link', 'shunt')
            ]
        )


def convert_matpower_to_heterodata():
    raise NotImplementedError(
        "TODO: add a utils function outside the class to convert a MATPOWER file to HeteroData format")
