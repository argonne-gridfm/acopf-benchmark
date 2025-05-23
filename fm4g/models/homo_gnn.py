"""
GNN models, How it is structured and types of GNN models: Transformer, GAT, GCN, GIN
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
from torch_geometric.nn import (GATConv, GCNConv, GINEConv, NNConv,
                                TransformerConv)
from torch_geometric.nn.pool import (global_add_pool, global_max_pool,
                                     global_mean_pool)

from fm4g.utils import from_adj_to_edge_index_torch


def get_gnnNets(input_dim, output_dim, model_params):
    r""" Get the GNN model based on the model name.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        model_params (dict): Model parameters.

    Returns:
        nn.Module: GNN model.
    """
    if model_params["model_name"].lower() in ["base", "gcn", "gat", "gin", "transformer"]:
        GNNmodel = model_params["model_name"].upper()
        return eval(GNNmodel)(input_dim=input_dim,
                              output_dim=output_dim,
                              model_params=model_params)
    else:
        raise ValueError(
            f"GNN name should be gcn, gat, gin or transformer " f"and {model_params.gnn_name} is not defined."
        )


def identity(x: torch.Tensor, batch: torch.Tensor):
    r"""Identity function for readout layer.

    Args:
        x (torch.Tensor): Node embeddings.
        batch (torch.Tensor): Batch vector.

    Returns:
        torch.Tensor: The node embeddings.
    """
    return x


def cat_max_sum(x, batch):
    r"""Concatenate max and sum of node embeddings.

    Args:
        x (torch.Tensor): Node embeddings.
        batch (torch.Tensor): Batch vector.

    Returns:
        torch.Tensor: Concatenated max and sum of node embeddings.
    """
    node_dim = x.shape[-1]
    bs = max(torch.unique(batch)) + 1
    num_node = int(x.shape[0] / bs)
    x = x.reshape(-1, num_node, node_dim)
    return torch.cat([x.max(dim=1)[0], x.sum(dim=1)], dim=-1)


# def get_readout_layers(readout):
#     r"""Get the readout layer based on the readout name.

#     Args:
#         readout (str): Readout layer name.

#     Returns:
#         nn.Module: Readout layer.
#     """
#     readout_func_dict = {
#         "mean": global_mean_pool,
#         "sum": global_add_pool,
#         "max": global_max_pool,
#         "identity": identity,
#         "cat_max_sum": cat_max_sum,
#     }
#     readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
#     return readout_func_dict[readout.lower()]


# class GNNPool(nn.Module):
#     def __init__(self, readout):
#         super().__init__()
#         self.readout = get_readout_layers(readout)

#     def forward(self, x, batch):
#         return self.readout(x, batch)


##
# GNN models
##
class GNNBase(nn.Module):
    r""" Homogeneous GNN Base model for all GNN models.

    Attributes:
        # TODO: add attr.
    """

    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r"""Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, "edge_attr"):
                    edge_attr = data.edge_attr
                else:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                if hasattr(data, "batch"):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if hasattr(data, "edge_weight") and data.edge_weight is not None:
                    edge_weight = data.edge_weight
                else:
                    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_attr = torch.ones(
                    (edge_index.shape[1], self.edge_dim),
                    dtype=torch.float32,
                    device=x.device,
                )
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            elif len(args) == 3:
                x, edge_index, edge_attr = args[0], args[1], args[2]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            elif len(args) == 4:
                x, edge_index, edge_attr, batch = args[0], args[1], args[2], args[3]
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
            else:
                raise ValueError(
                    f"forward's args should take 1, 2 or 3 arguments but got {len(args)}"
                )
        else:
            data: Batch = kwargs.get("data")
            if not data:
                x = kwargs.get("x")
                edge_index = kwargs.get("edge_index")
                adj = kwargs.get("adj")
                edge_weight = kwargs.get("edge_weight")
                if "edge_index" not in kwargs:
                    assert (
                        adj is not None
                    ), "forward's args is empty and required adj is not in kwargs"
                    if torch.is_tensor(adj):
                        edge_index, edge_weight = from_adj_to_edge_index_torch(adj)
                    else:
                        edge_index, edge_weight = from_adj_to_edge_index_torch(torch.from_numpy(adj))
                if "adj" not in kwargs:
                    assert (
                        edge_index is not None
                    ), "forward's args is empty and required edge_index is not in kwargs"
                assert (
                    x is not None
                ), "forward's args is empty and required node features x is not in kwargs"
                edge_attr = kwargs.get("edge_attr")
                if "edge_attr" not in kwargs:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                batch = kwargs.get("batch")
                if torch.is_tensor(batch):
                    if batch.size == 0:
                        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                else:
                    if not batch:
                        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if "edge_weight" not in kwargs:
                    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, "edge_attr"):
                    edge_attr = data.edge_attr
                    if edge_attr is None:
                        edge_attr = torch.ones(
                            (edge_index.shape[1], self.edge_dim),
                            dtype=torch.float64,
                            device=x.device,
                        )
                else:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                if hasattr(data, "batch"):
                    batch = data.batch
                    if batch is None:
                        batch = torch.zeros(
                            x.shape[0], dtype=torch.int64, device=x.device
                        )
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if hasattr(data, "edge_weight"):
                    edge_weight = data.edge_weight
                    if edge_weight is None:
                        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
                else:
                    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
        return x, edge_index, edge_attr, edge_weight, batch


# Basic structure of GNNs
class GNN_basic(GNNBase):
    def __init__(self,
                 input_dim,
                 output_dim,
                 model_params,
                 ):
        super(GNN_basic, self).__init__()  # edge_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = model_params["edge_dim"]
        self.num_layers = model_params["num_layers"]
        self.hidden_dim = model_params["hidden_dim"]
        self.dropout = model_params["dropout"]
        # readout
        self.readout = model_params["readout"]
        # self.readout_layer = GNNPool(self.readout)
        # self.default_num_nodes = model_params["default_num_nodes"]
        self.get_layers()

    def get_layers(self):
        # GNN layers
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for layer in range(self.num_layers):
            self.convs.append(NNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim * 2 if self.readout == 'cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return

    def forward(self, *args, **kwargs):
        r""" Forward pass of GNN model.
        """
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        # logits of ACOPF variables
        self.logits = self.mlps(emb)
        return self.logits

    def loss(self, pred, label):
        r""" Mean Squared Error Loss

        Args:
            pred (torch.Tensor): Predicted values.
            label (torch.Tensor): True values.

        Returns:
            torch.Tensor: Mean Squared Error Loss.
        """
        return F.mse_loss(pred, label)

    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        for layer in self.convs:
            x = layer(x, edge_index, edge_attr * edge_weight[:, None])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    # def get_graph_rep(self, *args, **kwargs):
    #     x, edge_index, edge_attr, edge_weight, batch = self._argsparse(*args, **kwargs)
    #     for layer in self.convs:
    #         x = layer(x, edge_index, edge_attr * edge_weight[:, None])
    #         x = F.relu(x)  # maybe replace the ReLU with LeakyReLU
    #         x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.readout_layer(x, batch)
    #     return x

    def get_pred_label(self, pred):
        return pred.argmax(dim=1)


class GAT(GNN_basic):
    def __init__(self, input_dim, output_dim, model_params):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for layer in range(self.num_layers):
            self.convs.append(
                GATConv(current_dim, self.hidden_dim, edge_dim=self.edge_dim)
            )
            current_dim = self.hidden_dim
        # FC layers

        mlp_dim = current_dim * 2 if self.readout == 'cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return


class GCN(GNN_basic):
    def __init__(self,
                 input_dim,
                 output_dim,
                 model_params):
        super().__init__(input_dim,
                         output_dim,
                         model_params)

    def get_layers(self):
        r""" Prepare the layers of GCN model. """
        # GCN layers
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for layer in range(self.num_layers):
            self.convs.append(GCNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim

        # MLP layers
        mlp_dim = current_dim * 2 if self.readout == 'cat_max_sum' else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return

    def get_emb(self, *args, **kwargs):
        r""" Get the node embeddings from GCN model. """
        x, edge_index, _, _, _ = self._argsparse(*args, **kwargs)

        for layer in self.convs:
            x = layer(x, edge_index)
            nn.PReLU()
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class GIN(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        print(current_dim)
        for layer in range(self.num_layers):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(current_dim, self.hidden_dim),
                        nn.ReLU(),
                        # nn.Linear(current_dim, self.hidden_dim),
                        # nn.PReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.Sigmoid(),
                    ),
                    edge_dim=self.edge_dim,
                )
            )
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim * 2 if self.readout == 'cat_max_sum' else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return


class TRANSFORMER(GNN_basic):  # uppercase
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for layer in range(self.num_layers):
            self.convs.append(
                TransformerConv(current_dim, self.hidden_dim, heads=4, edge_dim=self.edge_dim, concat=False)
            )
            current_dim = self.hidden_dim * 1

        # FC layers
        mlp_dim = current_dim * 2 if self.readout == 'cat_max_sum' else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return
