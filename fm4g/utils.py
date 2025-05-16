import argparse
import os
import random

import numpy as np
import torch


def from_adj_to_edge_index_torch(adj):
    r""" Convert a dense adjacency matrix to a sparse edge index and edge attribute tensor.
    The edge attribute tensor is the non-zero values (weights) of the adjacency matrix.

    Args:
        adj (torch.Tensor): Dense adjacency matrix.

    Returns:
        edge_index (torch.Tensor): Sparse edge index tensor.
        edge_attr (torch.Tensor): Edge attribute tensor
    """
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices().to(dtype=torch.long)
    edge_attr = adj_sparse.values()
    return edge_index, edge_attr


def check_dir(dir_path):
    r""" Check if the directory exists, if not create it.

    Args:
        dir_path (str): Directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_args_group(parser, args):
    r""" Create a dictionary of arguments grouped by the argument group title.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
        args (argparse.Namespace): Arguments.
    """
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = group_dict
    return arg_groups


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


def get_data_args(dataset, args):
    r""" Get dataset arguments.

    Args:
        dataset (torch_geometric.data.Data): Dataset.
        args (argparse.Namespace): Arguments.

    Returns:
        args (argparse.Namespace): Arguments.
    """
    args.num_classes = dataset.data.y.size(1)
    args.num_node_features = dataset.data.x.size(1)

    if dataset.data.edge_attr.ndim == 1:
        dataset.data.edge_attr = torch.unsqueeze(dataset.data.edge_attr, 1)

    args.edge_dim = dataset.data.edge_attr.size(1)
    return args


def arg_parse():
    r""" Parse args for training for Homogeneous Graph Neural Networks. """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_save_dir",
                        type=str,
                        default="/home/jinh/tmp/data",
                        help="Directory where benchmark is located")
    parser.add_argument("--logs_save_dir",
                        type=str,
                        default="/home/jinh/tmp/logs",
                        help="Directory where logs are saved")
    parser.add_argument("--model_save_dir",
                        type=str,
                        default="/home/jinh/tmp/modeltest",
                        help="saving directory for gnn model")
    parser.add_argument("--mask_save_dir",
                        type=str,
                        default="/home/jinh/tmp/masks",
                        help="Directory where masks are saved")
    parser.add_argument("--result_save_dir",
                        type=str,
                        default="/home/jinh/tmp/results",
                        help="Directory where results are saved")
    parser.add_argument("--fig_save_dir",
                        type=str,
                        default="/home/jinh/tmp/figures",
                        help="Directory where figures are saved")

    # dataset parameters
    parser_dataset_params = parser.add_argument_group("dataset_params")
    parser_dataset_params.add_argument("--dataset_name",
                                       type=str,
                                       default="pglib_opf_case14_ieee",
                                       help="dataset name")
    parser_dataset_params.add_argument("--seed",
                                       type=int,
                                       default=42,
                                       help="random seed")
    parser_dataset_params.add_argument("--train_ratio",
                                       type=float,
                                       default=0.8,
                                       help="train ratio")
    parser_dataset_params.add_argument("--test_ratio",
                                       type=float,
                                       default=0.1,
                                       help="test ratio")
    parser_dataset_params.add_argument("--val_ratio",
                                       type=float,
                                       default=0.1,
                                       help="validation ratio")
    parser_dataset_params.add_argument("--random_split_flag",
                                       action='store_true',
                                       help="Flag to enable random split")

    # optimization parameters
    parser_optimizer_params = parser.add_argument_group("optimizer_params")
    parser_optimizer_params.add_argument("--lr",
                                         type=float,
                                         default=0.001,
                                         help="learning rate")
    parser_optimizer_params.add_argument("--weight_decay",
                                         type=float,
                                         default=5e-5,
                                         help="weight decay")

    # training parameters
    parser_train_params = parser.add_argument_group("train_params")
    parser_train_params.add_argument("--num_epochs",
                                     type=int,
                                     default=30,
                                     help="Number of epochs to train.")
    parser_train_params.add_argument("--num_early_stop",
                                     type=int,
                                     default=10,
                                     help="Num steps before stopping")

    # model parameters
    parser_model_params = parser.add_argument_group("model_params")
    parser_model_params.add_argument("--model_name",
                                     type=str,
                                     default="transformer",
                                     choices=["gat", "gcn", "gin", "transformer"],
                                     help="Model name. GCN can only be used for data with no or 1D edge features.")
    parser_model_params.add_argument("--batch_size",
                                     help="batch_size",
                                     type=int,
                                     default=16)
    parser_model_params.add_argument("--hidden_dim",
                                     type=int,
                                     default=32,
                                     help="Hidden dimension")
    parser_model_params.add_argument("--num_layers",
                                     type=int,
                                     default=3,
                                     help="Number of layers before pooling")
    parser_model_params.add_argument("--dropout",
                                     type=float,
                                     default=0.1,
                                     help="Dropout rate.")
    parser_model_params.add_argument("--readout",
                                     type=str,
                                     default="mean",
                                     choices=["mean", "sum", "max"],
                                     help="Readout type.")
    parser_model_params.add_argument("--edge_dim",
                                     type=int,
                                     default=2,
                                     help="Edge feature dimension (only for GAT, GIN and TRANSFORMER model).")

    args = parser.parse_args()
    return args
