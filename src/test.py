# test.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from ingestion import prepare_datasets
from model import GAT  # Import the model here
import copy
from sklearn.model_selection import train_test_split

def test_model(

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base directory where your data is stored
base_dir = os.getcwd()

# Prepare datasets
graphs_sc, labels_sc, graphs_sc_combined, labels_sc_combined, feature_names = prepare_datasets(base_dir)

# Choose which dataset to use (graphs_sc or graphs_sc_combined)
graphs = graphs_sc_combined  # Use combined features
labels = labels_sc_combined

# Load the best model and normalization parameters
checkpoint = torch.load('best_model.pth', map_location=device)
config = checkpoint['config']

# Initialize the model
model = GAT(
    in_channels=config["in_channels"],
    out_channels=config["out_channels"],
    num_heads=config["num_heads"],
    num_classes=config["num_classes"]
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Preprocess the data using the saved normalization parameters
def normalize_features_test(graphs, mean, std):
    def apply_normalization(graphs):
        normalized_graphs = []
        for graph in graphs:
            normalized_x = (graph.x - mean) / (std + 1e-8)
            normalized_graph = copy.deepcopy(graph)
            normalized_graph.x = normalized_x
            normalized_graphs.append(normalized_graph)
        return normalized_graphs

    normalized_graphs = apply_normalization(graphs)
    return normalized_graphs

def scale_edge_weights_test(graphs, min_weight, max_weight):
    def apply_scaling(graphs):
        scaled_graphs = []
        for graph in graphs:
            scaled_weights = (graph.edge_attr - min_weight) / (max_weight - min_weight + 1e-8)
            scaled_graph = copy.deepcopy(graph)
            scaled_graph.edge_attr = scaled_weights
            scaled_graphs.append(scaled_graph)
        return scaled_graphs

    scaled_graphs = apply_scaling(graphs)
    return scaled_graphs

# Apply normalization and scaling
normalized_graphs = normalize_features_test(graphs, train_feature_mean, train_feature_std)
scaled_graphs = scale_edge_weights_test(normalized_graphs, min_edge_weight, max_edge_weight)

# Create the test dataset
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

test_dataset = GraphDataset(scaled_graphs, labels)

# Define the collate function
def collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

# Create data loader
test_loader = GraphDataLoader(
    test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Test the model
test_loss, test_accuracy, test_f1 = test(test_loader, model, criterion, device)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')
