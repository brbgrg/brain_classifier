# data_utils.py

import torch
import copy
import dgl
from torch.utils.data import Dataset

def normalize_features(graphs, mean=None, std=None):
    """
    Normalize node features. If mean and std are provided, use them;
    otherwise, compute from the graphs.
    Returns normalized graphs, mean, and std.
    """
    if mean is None or std is None:
        # Compute mean and std from all graphs
        all_features = torch.cat([graph.x for graph in graphs], dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)

    normalized_graphs = []
    for graph in graphs:
        normalized_x = (graph.x - mean) / (std + 1e-8)
        normalized_graph = copy.deepcopy(graph)
        normalized_graph.x = normalized_x
        normalized_graphs.append(normalized_graph)
    return normalized_graphs, mean, std

def scale_edge_weights(graphs, min_weight=None, max_weight=None):
    """
    Scale edge weights. If min_weight and max_weight are provided, use them;
    otherwise, compute from the graphs.
    Returns scaled graphs, min_weight, and max_weight.
    """
    if min_weight is None or max_weight is None:
        # Compute min and max from all graphs
        all_edge_weights = torch.cat([graph.edge_attr for graph in graphs], dim=0)
        min_weight = all_edge_weights.min().item()
        max_weight = all_edge_weights.max().item()

    scaled_graphs = []
    for graph in graphs:
        scaled_weights = (graph.edge_attr - min_weight) / (max_weight - min_weight + 1e-8)
        scaled_graph = copy.deepcopy(graph)
        scaled_graph.edge_attr = scaled_weights
        scaled_graphs.append(scaled_graph)
    return scaled_graphs, min_weight, max_weight

def to_dgl(pyg_data):
    """Convert PyTorch Geometric Data to DGLGraph."""
    g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]))
    g.ndata['feat'] = pyg_data.x
    g.edata['weight'] = pyg_data.edge_attr
    return g

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = [to_dgl(graph) for graph in graphs]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        return graph, label

def collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels
