import copy
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset
import torch

# Preprocessing post-split

def normalize_features(train_graphs, val_graphs, test_graphs):
    def apply_normalization(graphs):
        normalized_graphs = []
        for graph in graphs:
            mean = graph.x.mean(dim=0)
            std = graph.x.std(dim=0)
            normalized_x = (graph.x - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
            normalized_graph = copy.deepcopy(graph)
            normalized_graph.x = normalized_x
            normalized_graphs.append(normalized_graph)
        return normalized_graphs

    normalized_train_graphs = apply_normalization(train_graphs)
    normalized_val_graphs = apply_normalization(val_graphs)
    normalized_test_graphs = apply_normalization(test_graphs)

    return normalized_train_graphs, normalized_val_graphs, normalized_test_graphs


# Scale the edge weights in the range [0, 1]

def scale_edge_weights(graphs):
    scaled_graphs = []
    for graph in graphs:
        edge_weights = graph.edge_attr
        min_weight = edge_weights.min().item()
        max_weight = edge_weights.max().item()
        scaled_weights = (edge_weights - min_weight) / (max_weight - min_weight)
        scaled_graph = copy.deepcopy(graph)
        scaled_graph.edge_attr = scaled_weights
        scaled_graphs.append(scaled_graph)
    return scaled_graphs


# Create a PyTorch dataset as a subclass of torch.utils.data.Dataset
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

def to_dgl(pyg_data):
    """Convert PyTorch Geometric Data to DGLGraph."""
    g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]))
    g.ndata['feat'] = pyg_data.x
    g.edata['weight'] = pyg_data.edge_attr
    return g

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = [to_dgl(graph) for graph in graphs]
        self.labels = labels

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