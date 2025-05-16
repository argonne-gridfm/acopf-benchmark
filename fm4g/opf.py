""" Customized OPFDataset class for the FM4G project.

Changes:
- Parallelize download and data processing tasks.
- Remove temporary files after processing.

Reference:
- [OPFDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.OPFDataset.html)
- [OPFData](https://arxiv.org/abs/2406.07234)

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import json
import os
import os.path as osp
import pickle
import random
import shutil
from glob import glob
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandapower as pp
import torch
import tqdm
from joblib import Parallel, delayed, parallel_backend
from pandapower.pypower.idx_brch import (BR_B, BR_R, BR_X, F_BUS, RATE_A,
                                         RATE_B, RATE_C, SHIFT, T_BUS, TAP)
from pandapower.pypower.idx_bus import (BASE_KV, BS, BUS_AREA, BUS_I, BUS_TYPE,
                                        GS, PD, QD, VA, VM, VMAX, VMIN, ZONE)
from pandapower.pypower.idx_cost import (COST, MODEL, NCOST, POLYNOMIAL,
                                         PW_LINEAR, SHUTDOWN, STARTUP)
from pandapower.pypower.idx_gen import (GEN_BUS, GEN_STATUS, MBASE, PG, PMAX,
                                        PMIN, QG, QMAX, QMIN, VG)
from torch import Tensor
from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_tar)
from torch_geometric.io import fs


class OPFDataset(InMemoryDataset):
    r"""The heterogeneous OPF data from the `"Large-scale Datasets for AC
    Optimal Power Flow with Topological Perturbations"
    <https://arxiv.org/abs/2406.07234>`_ paper.

    :class:`OPFDataset` is a large-scale dataset of solved optimal power flow
    problems, derived from the
    `pglib-opf <https://github.com/power-grid-lib/pglib-opf>`_ dataset.

    The physical topology of the grid is represented by the :obj:`"bus"` node
    type, and the connecting AC lines and transformers. Additionally,
    :obj:`"generator"`, :obj:`"load"`, and :obj:`"shunt"` nodes are connected
    to :obj:`"bus"` nodes using a dedicated edge type each, *e.g.*,
    :obj:`"generator_link"`.

    Edge direction corresponds to the properties of the line, *e.g.*,
    :obj:`b_fr` is the line charging susceptance at the :obj:`from`
    (source/sender) bus.

    Args:
        root (str): Root directory where the dataset should be saved.
        case_name (str, optional): The name of the original pglib-opf case.
            (default: :obj:`"pglib_opf_case14_ieee"`)
        num_groups (int, optional): The dataset is divided into 20 groups with
            each group containing 15,000 samples.
            For large networks, this amount of data can be overwhelming.
            The :obj:`num_groups` parameters controls the amount of data being
            downloaded. Allowed values are :obj:`[1, 20]`.
            (default: :obj:`20`)
        topological_perturbations (bool, optional): Whether to use the dataset
            with added topological perturbations. (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in
            a :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes
            in a :obj:`torch_geometric.data.HeteroData` object and returns
            a transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :obj:`torch_geometric.data.HeteroData` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = 'https://storage.googleapis.com/gridopt-dataset'

    def __init__(
        self,
        root: str,
        case_name: Literal[
            'pglib_opf_case14_ieee',
            'pglib_opf_case30_ieee',
            'pglib_opf_case57_ieee',
            'pglib_opf_case118_ieee',
            'pglib_opf_case500_goc',
            'pglib_opf_case2000_goc',
            'pglib_opf_case4661_sdet',
            'pglib_opf_case6470_rte',
            'pglib_opf_case10000_goc',
            'pglib_opf_case13659_pegase',
        ] = 'pglib_opf_case14_ieee',
        num_groups: int = 20,
        # groups: Union[str, List[int]] = "all",
        topological_perturbations: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        save_pkl: bool = False,
        force_reload: bool = False,
        # raw_folder: str = None,
    ) -> None:

        self.case_name = case_name
        # self.groups = groups
        self.num_groups = num_groups
        self.topological_perturbations = topological_perturbations

        self._release = 'dataset_release_1'
        if topological_perturbations:
            self._release += '_nminusone'
        self.save_pkl = save_pkl

        # # # NOTE: add admittance matrix Y to the dataset
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # matpower_file = osp.join(current_dir, "../data/pglib", f"{case_name}.m")
        # net = pp.converter.from_mpc(matpower_file)
        # ppc = pp.converter.pypower.to_ppc(net, init='flat')
        # self.Y, _, _ = pp.makeYbus_pypower(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        # self.Y_real = torch.tensor(self.Y.real.todense(), dtype=torch.float32)
        # self.Y_imag = torch.tensor(self.Y.imag.todense(), dtype=torch.float32)
        # self.load_bus_indices = net.load.bus.values.astype(np.int32)

        # self.raw_folder = raw_folder
        # NOTE:
        #   check downloaded: if raw files are not ready, download from url
        #   check processed: if processed files are not exist, process raw files
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        # DEBUG: return the random sampled .pt file(s) @cshjin
        # # Number of files to randomly select
        # k = 5
        # selected_files = random.sample(self.processed_paths, k)

        # data_list = []
        # for file in selected_files:
        #     data, slices = torch.load(file)
        #     data_list.append(data)

        # self.data, self.slices = self.collate(data_list)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices = out

    @property
    def raw_dir(self) -> str:
        r""" Raw data folder. """
        return osp.join(self.root, self._release, self.case_name, 'raw')

    @property
    def processed_dir(self) -> str:
        r""" Processed data folder. """
        return osp.join(self.root, self._release, self.case_name, "processed")

    @property
    def tmp_dir(self) -> str:
        # NOTE: new class attribute
        r""" Temporary data folder. """
        return osp.join(self.raw_dir,
                        "gridopt-dataset-tmp",
                        self._release,
                        self.case_name)

    @property
    def raw_file_names(self) -> List[str]:
        r""" Raw file names, which are stored locally. """
        return [f'{self.case_name}_{i}.tar.gz' for i in range(self.num_groups)]

    @property
    def processed_file_names(self) -> List[str]:
        # TODO: add sampling options to load specific group file(s)
        return [f'group_{idx}.pt' for idx in range(self.num_groups)]

    def download(self) -> None:
        r""" Download .tar.gz files """
        print("download files")

        results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self.download_and_extract)(name)
            for name in self.raw_file_names)

        print(f"Downloaded {len(results)} files")

    def download_and_extract(self, name: str) -> None:
        r""" Download and extract a .tar.gz file. """
        url = f'{self.url}/{self._release}/{name}'
        path = download_url(url, self.raw_dir)
        extract_tar(path, self.raw_dir)

    def process(self) -> None:
        r""" Process the raw files into a single file. """
        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        with parallel_backend('multiprocessing'):
            results = Parallel(n_jobs=-1)(
                delayed(self.process_group)(group) for group in range(self.num_groups))

        print(f"Processed {len(results)} groups")

        # NOTE: remove tmp_dir content to save local space
        shutil.rmtree(osp.join(self.raw_dir, 'gridopt-dataset-tmp'))

    def process_group(self, group_id: int):
        r""" Process a single group of files, save processed data to disk.

        Args:
            group_id (int): Group id.
        """

        group_json_files = glob(osp.join(self.tmp_dir, f'group_{group_id}', '*.json'))
        #
        if len(group_json_files) < 15000:
            extract_tar(osp.join(self.raw_dir, self.raw_file_names[group_id]), self.raw_dir)
            group_json_files = glob(osp.join(self.tmp_dir, f'group_{group_id}', '*.json'))

        data_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(process_json_file)(fn) for fn in tqdm.tqdm(group_json_files, desc=f"Group {group_id}"))

        if self.pre_filter is not None or self.pre_transform is not None:
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

        torch.save((self._data, self.slices), osp.join(self.processed_dir, f'group_{group_id}.pt'))

        # # DEPRECATED: save data_list to HDF5 file format.
        # with h5py.File(osp.join(self.processed_dir, f'group_{group_id}.h5'), 'w') as h5f:
        #     for i, data in enumerate(data_list):
        #         grp = h5f.create_group(f'data_{i}')
        #         for key, value in data.items():
        #             grp.create_dataset(key, data=value.numpy())

        # NOTE: save data_list to pickle file format.
        if self.save_pkl:
            with open(osp.join(self.processed_dir, f'group_{group_id}.pkl'), 'wb') as f:
                pickle.dump(data_list, f)

    def combine_datasets(self, file_paths: List[str]) -> List[HeteroData]:
        r""" Combine datasets from multiple files. """
        # TODO: double check the correctness of this function
        combined_data = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            combined_data.extend(data)
        return combined_data

    def merge_group_files(self) -> None:
        r"""Merge group files into single train/val/test based on groups options. """
        group_ids = range(self.num_groups)
        data_files = [osp.join(self.processed_dir, f'group_{i}.pkl') for i in group_ids]
        combined_data = self.combine_datasets(data_files)
        self.data, self.slices = self.collate(combined_data)

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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'case_name={self.case_name}, '
                f'topological_perturbations={self.topological_perturbations})')


def extract_edge_index(obj: Dict, edge_name: str) -> Tensor:
    # TODO: convert the list to np.ndarray before converting to torch.Tensor
    return torch.tensor(np.array([
        obj['grid']['edges'][edge_name]['senders'],
        obj['grid']['edges'][edge_name]['receivers'],
    ]))


def extract_edge_index_rev(obj: Dict, edge_name: str) -> Tensor:
    return torch.tensor(np.array([
        obj['grid']['edges'][edge_name]['receivers'],
        obj['grid']['edges'][edge_name]['senders'],
    ]))


def process_json_file(json_file):
    r"""Process a single json file.

    Args:
        json_file (str): Path to the json file.

    Returns:
        data (HeteroData): Processed single data object.
    """
    with open(json_file) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {json_file}")
            return None

    grid = obj['grid']
    solution = obj['solution']
    metadata = obj['metadata']

    # Graph-level properties:
    data = HeteroData()
    data.baseMVA = torch.tensor(grid['context']).view(-1).item()
    data.objective = torch.tensor(metadata['objective'])

    # Nodes (only some have a target):
    data['bus'].x = torch.tensor(grid['nodes']['bus'])
    data['bus'].y = torch.tensor(solution['nodes']['bus'])

    data['generator'].x = torch.tensor(grid['nodes']['generator'])
    data['generator'].y = torch.tensor(solution['nodes']['generator'])

    data['load'].x = torch.tensor(grid['nodes']['load'])

    data['shunt'].x = torch.tensor(grid['nodes']['shunt'])

    # Edges (only ac lines and transformers have features):
    data['bus', 'ac_line', 'bus'].edge_index = extract_edge_index(obj, 'ac_line')
    data['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(grid['edges']['ac_line']['features'])
    data['bus', 'ac_line', 'bus'].edge_label = torch.tensor(solution['edges']['ac_line']['features'])

    data['bus', 'transformer', 'bus'].edge_index = extract_edge_index(obj, 'transformer')
    data['bus', 'transformer', 'bus'].edge_attr = torch.tensor(grid['edges']['transformer']['features'])
    data['bus', 'transformer', 'bus'].edge_label = torch.tensor(solution['edges']['transformer']['features'])

    data['generator', 'generator_link', 'bus'].edge_index = extract_edge_index(obj, 'generator_link')
    data['bus', 'generator_link', 'generator'].edge_index = extract_edge_index_rev(obj, 'generator_link')

    data['load', 'load_link', 'bus'].edge_index = extract_edge_index(obj, 'load_link')
    data['bus', 'load_link', 'load'].edge_index = extract_edge_index_rev(obj, 'load_link')

    data['shunt', 'shunt_link', 'bus'].edge_index = extract_edge_index(obj, 'shunt_link')
    data['bus', 'shunt_link', 'shunt'].edge_index = extract_edge_index_rev(obj, 'shunt_link')

    return data


def process_matpower_file(matpower_file):
    r"""Process a single MATPOWER case file.

    Args:
        matpower_file (str): Path to the MATPOWER case file.

    Returns:
        data (HeteroData): Processed single data object.
    """

    net = pp.converter.from_mpc(matpower_file)
    ppc = pp.converter.pypower.to_ppc(net, init='flat')
    # pp.runpp(net, numba=False)

    # Calculate the admittance matrix Y
    # NOTE: https://matpower.org/docs/ref/matpower5.0/makeYbus.html
    Y_sp, _, _ = pp.makeYbus_pypower(ppc['baseMVA'], ppc['bus'], ppc['branch'])

    load_bus_indices = net.load.bus.values.astype(np.int32)
    gen_bus_indices = net.gen.bus.values.astype(np.int32)

    obj = {
        'grid': {
            'context': [[ppc['baseMVA']]],
            'nodes': {
                'bus': ppc['bus'][:, [BUS_TYPE, BASE_KV, VMAX, VMIN]].astype(np.float32),
                'generator': ppc['gen'][:, [QMAX, QMIN, VG, MBASE, PMAX, PMIN]].astype(np.float32),
                'gencost': np.pad(ppc['gencost'], ((0, 0), (0, 7 - ppc['gencost'].shape[1])), 'constant', constant_values=0).astype(np.float32),
                'load': ppc['bus'][:, [PD, QD]].astype(np.float32),
                'shunt': ppc['bus'][:, [GS, BS]].astype(np.float32)
            },
            'edges': {
                'ac_line': {
                    'senders': ppc['branch'][:, F_BUS].astype(int),
                    'receivers': ppc['branch'][:, T_BUS].astype(int),
                    # TAP, SHIFT are at indices 8, 9
                    'features': ppc['branch'][:, [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT]].astype(np.float32)
                },
                'transformer': {
                    'senders': ppc['branch'][ppc['branch'][:, 8] != 0][:, F_BUS].astype(int),
                    'receivers': ppc['branch'][ppc['branch'][:, 8] != 0][:, T_BUS].astype(int),
                    'features': ppc['branch'][ppc['branch'][:, 8] != 0][:, [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT]].astype(np.float32)
                },
                'generator_link': {
                    'senders': np.arange(ppc['gen'].shape[0]),
                    'receivers': ppc['gen'][:, GEN_BUS].astype(int)
                },
                'load_link': {
                    'senders': ppc['bus'][:, BUS_I].astype(int),
                    'receivers': ppc['bus'][:, BUS_I].astype(int)
                },
                'shunt_link': {
                    'senders': ppc['bus'][:, BUS_I].astype(int),
                    'receivers': ppc['bus'][:, BUS_I].astype(int)
                }
            }
        },
        'metadata': {
            'objective': 0.0  # not given from MATPOWER file
        }
    }
    grid = obj['grid']

    # Graph-level properties:
    data = HeteroData()
    data.Y = Y_sp
    # Store baseMVA explicitly, remove generic data.x
    # data.x = torch.tensor(grid['context']).view(-1)
    data.base_mva = ppc['baseMVA']
    data.load_bus_indices = load_bus_indices
    data.gen_bus_indices = gen_bus_indices
    # data.baseMVA = ppc['baseMVA'] # Already stored as data.base_mva
    # DEBUG: duplicated baseMVA
    data.baseMVA = ppc['baseMVA']

    # Nodes (only some have a target):
    data['bus'].x = torch.tensor(grid['nodes']['bus'])
    data['generator'].x = torch.tensor(grid['nodes']['generator'])
    data['gencost'].x = torch.tensor(grid['nodes']['gencost'])
    data['load'].x = torch.tensor(grid['nodes']['load'])
    data['shunt'].x = torch.tensor(grid['nodes']['shunt'])

    # Edges (only ac lines and transformers have features):
    data['bus', 'ac_line', 'bus'].edge_index = extract_edge_index(obj, 'ac_line')
    data['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(grid['edges']['ac_line']['features'])
    # data['bus', 'ac_line', 'bus'].edge_label = torch.tensor(solution['edges']['ac_line']['features'])

    data['bus', 'transformer', 'bus'].edge_index = extract_edge_index(obj, 'transformer')
    data['bus', 'transformer', 'bus'].edge_attr = torch.tensor(grid['edges']['transformer']['features'])
    # data['bus', 'transformer', 'bus'].edge_label = torch.tensor(solution['edges']['transformer']['features'])
    # print("Warning: edge_label is not available")

    data['generator', 'generator_link', 'bus'].edge_index = extract_edge_index(obj, 'generator_link')
    data['bus', 'generator_link', 'generator'].edge_index = extract_edge_index_rev(obj, 'generator_link')

    data['load', 'load_link', 'bus'].edge_index = extract_edge_index(obj, 'load_link')
    data['bus', 'load_link', 'load'].edge_index = extract_edge_index_rev(obj, 'load_link')

    data['shunt', 'shunt_link', 'bus'].edge_index = extract_edge_index(obj, 'shunt_link')
    data['bus', 'shunt_link', 'shunt'].edge_index = extract_edge_index_rev(obj, 'shunt_link')

    return data
