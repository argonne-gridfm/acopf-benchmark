"""
Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (MLP, GATConv, GCNConv, GINConv, GraphConv,
                                HeteroConv, HGTConv, Linear, RGATConv,
                                SAGEConv)


class HeteroGNN(torch.nn.Module):
    def __init__(
            self,
            metadata,
            input_channels=None,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            backend="sage",
            **kwargs):
        r""" Heterogeneous Graph Neural Network (HeteroGNN) model.

        Args:
            metadata (tuple): Tuple of node types and edge types.
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
        """
        super(HeteroGNN, self).__init__()
        self.lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        edge_types = metadata[1]
        self.backend = backend
        self.edge_attr_support = backend == "gat"

        # Input layers for each node type
        for node_type in node_types:
            if input_channels is not None:
                self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)
            else:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # Heterogeneous graph convolutional layers for edges
        self.convs = torch.nn.ModuleList()
        conv_layer_map = {
            "sage": lambda: SAGEConv((-1, -1), hidden_channels),
            "gcn": lambda: GraphConv((-1, -1), hidden_channels),
            "gin": lambda: GINConv(MLP([hidden_channels, hidden_channels])),
            "gat": lambda: GATConv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=-1 if self.edge_attr_support else None),
        }

        if backend not in conv_layer_map:
            raise ValueError(f"Unknown backend: {backend}")

        for _ in range(num_layers - 1):
            conv = HeteroConv({
                edge_type: conv_layer_map[backend]()
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Output layers for target node types, ACOPF variables
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            for rel_conv in conv.convs.values():
                if hasattr(rel_conv, 'reset_parameters'):
                    rel_conv.reset_parameters()
            conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Transform input features
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Message passing
        for conv in self.convs:
            if self.edge_attr_support and edge_attr_dict is not None:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            # NOTE: no activation function applied here
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        # Apply final activation sigmoid to ensure values range from 0 to 1
        # bus_out = torch.sigmoid(bus_out)
        # gen_out = torch.sigmoid(gen_out)
        bus_out[:, 1] = torch.relu(bus_out[:, 1])  # Ensure voltage magnitude is non-negative
        return {"bus": bus_out, "generator": gen_out}


class RGAT(torch.nn.Module):

    def __init__(self,
                 metadata,
                 input_channels=None,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 num_heads=1,
                 **kwargs):
        r""" Relational Graph Attention Network (RGAT) model.

        Args:
            metadata (tuple): Tuple of node types and edge types.
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_heads (int): Number of multi-head-attention heads. Defaults to 1.
            num_layers (int): Number of layers. Defaults to 3.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        edge_types = metadata[1]

        for node_type in node_types:
            if input_channels is not None:
                self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)
            else:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # TODO: verify the RGATConv
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = RGATConv(metadata, hidden_channels, num_relations=3)
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Transform input features
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        # Apply final activation sigmoid to ensure values range from 0 to 1
        bus_out = torch.sigmoid(bus_out)
        gen_out = torch.sigmoid(gen_out)
        return {"bus": bus_out, "generator": gen_out}


class HEAT(torch.nn.Module):
    def __init__(
            self,
            metadata,
            input_channels=None,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            attention_heads=1,
            **kwargs):
        r""" Heterogeneous Edge-Attributed Transformer (HEAT) model.

        Args:
            metadata (tuple): Tuple of node types and edge types.
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            attention_heads (int): Number of attention heads. Defaults to 1.
        """
        super().__init__()

        from torch_geometric.nn import HEATConv  # Import HEATConv

        self.lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        edge_types = metadata[1]

        # Input layers for each node type
        for node_type in node_types:
            if input_channels is not None:
                self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)
            else:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # TODO: HEATConv layers for handling heterogeneous edge attributes
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: HEATConv(
                    in_channels=(-1, -1),
                    out_channels=hidden_channels,
                    heads=attention_heads,
                    edge_dim=-1,  # Will use the edge attributes dimension from data
                    add_self_loops=False
                )
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Transform input features
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Message passing with edge attributes
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        # Apply final activation sigmoid to ensure values range from 0 to 1
        bus_out = torch.sigmoid(bus_out)
        gen_out = torch.sigmoid(gen_out)
        return {"bus": bus_out, "generator": gen_out}


class HGT(torch.nn.Module):

    def __init__(self,
                 metadata,
                 input_channels=None,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 num_heads=1,
                 **kwargs):
        r""" Heterogeneous Graph Transformer (HGT) model.

        Args:
            metadata (tuple): Tuple of node types and edge types.
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_heads (int): Number of multi-head-attention heads. Defaults to 1.
            num_layers (int): Number of layers. Defaults to 3.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        for node_type in self.node_types:
            if input_channels is not None:
                self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)
            else:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

    def forward(self, x_dict, edge_index_dict):
        # Transform input features
        # x_dict = {
        #     node_type: self.lin_dict[node_type](x).relu_()
        #     for node_type, x in x_dict.items()
        # }
        x_dict = {
            node_type: self.lin_dict[node_type](x_dict[node_type]).relu_()
            for node_type in self.node_types
        }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        # Apply final activation sigmoid to ensure values range from 0 to 1
        bus_out = torch.sigmoid(bus_out)
        gen_out = torch.sigmoid(gen_out)
        return {"bus": bus_out, "generator": gen_out}


class HEAT_v2(torch.nn.Module):
    def __init__(
            self,
            metadata,
            input_channels=None,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            attention_heads=1,
            edge_type_emb_dim=16,
            edge_attr_emb_dim=16,
            **kwargs):
        r""" Improved Heterogeneous Edge-Attributed Transformer (HEAT) model.

        Args:
            metadata (tuple): Tuple of node types and edge types.
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            attention_heads (int): Number of attention heads. Defaults to 1.
            edge_type_emb_dim (int): Dimension of edge type embeddings. Defaults to 16.
            edge_attr_emb_dim (int): Dimension of edge attribute embeddings. Defaults to 16.
        """
        super().__init__()

        from torch_geometric.nn import HEATConv

        self.lin_dict = torch.nn.ModuleDict()
        self.edge_lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        edge_types = metadata[1]

        # Input layers for nodes
        for node_type in node_types:
            if input_channels is not None:
                self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)
            else:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # Input layers for edges
        for edge_type in edge_types:
            # Convert edge_type tuple to string for ModuleDict key
            edge_type_str = str(edge_type)
            self.edge_lin_dict[edge_type_str] = Linear(-1, hidden_channels)

        # HEATConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # conv = HEATConv(
            #     in_channels=(-1, -1),
            #     out_channels=hidden_channels,
            #     heads=attention_heads,
            #     edge_dim=hidden_channels,
            #     add_self_loops=False
            # )
            conv = HEATConv(
                in_channels=-1,
                out_channels=hidden_channels,
                num_node_types=len(node_types),
                num_edge_types=len(edge_types),
                edge_type_emb_dim=edge_type_emb_dim,
                edge_dim=9,
                edge_attr_emb_dim=edge_attr_emb_dim,
                heads=attention_heads,
                concat=True,
            )
            self.convs.append(conv)

        # Output layers
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Transform node features
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Transform edge features
        edge_attr_dict = {
            edge_type: self.edge_lin_dict[edge_type](edge_attr).relu_()
            for edge_type, edge_attr in edge_attr_dict.items()
        }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        # Apply final activation sigmoid to ensure values range from 0 to 1
        bus_out = torch.sigmoid(bus_out)
        gen_out = torch.sigmoid(gen_out)
        return {"bus": bus_out, "generator": gen_out}


class HGNN_Base(torch.nn.Module):
    r""" Base class for Heterogeneous Graph Neural Network (HGNN) models. """

    def __init__(self, metadata, hidden_channels, out_channels=2, **kwargs):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        edge_types = metadata[1]

        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        raise NotImplementedError("Forward method not implemented in base class")
