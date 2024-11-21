# ingestion.py

import os
import numpy as np
import scipy.io
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.stats
import copy

def load_mat_file(path):
    """Load a .mat file and return the loaded data."""
    data = scipy.io.loadmat(path)
    return data

def load_all_data(base_dir):
    """Load connectivity data and features from .mat files."""
    # Paths to data files
    data_path = os.path.join(base_dir, "new_data", "scfc_schaefer100_ya_oa.mat")
    mod_deg_zscore_path = os.path.join(base_dir, "new_data", "mod_deg_zscore_scfc.mat")
    part_coeff_path = os.path.join(base_dir, "new_data", "part_coeff_scfc.mat")

    # Load data
    data = load_mat_file(data_path)
    mod_deg_zscore_data = load_mat_file(mod_deg_zscore_path)
    part_coeff_data = load_mat_file(part_coeff_path)

    # Extract connectivity matrices
    data_content = data['data'][0, 0]
    matrices = {
        'sc_ya': np.array(data_content['sc_ya']),
        'fc_ya': np.array(data_content['fc_ya']),
        'sc_oa': np.array(data_content['sc_oa']),
        'fc_oa': np.array(data_content['fc_oa'])
    }

    # Extract age data
    ages = {
        'age_ya': np.array(data_content['age_ya']).flatten(),
        'age_oa': np.array(data_content['age_oa']).flatten()
    }

    # Load modular degree z-score and participation coefficient
    mod_deg_zscore = {
        'fc_ya': np.array(mod_deg_zscore_data['mdz_fc'][:, :101]),
        'fc_oa': np.array(mod_deg_zscore_data['mdz_fc'][:, 101:]),
        'sc_ya': np.array(mod_deg_zscore_data['mdz_sc'][:, :101]),
        'sc_oa': np.array(mod_deg_zscore_data['mdz_sc'][:, 101:])
    }

    part_coeff = {
        'fc_ya': np.array(part_coeff_data['pc_fc'][:, :101]),
        'fc_oa': np.array(part_coeff_data['pc_fc'][:, 101:]),
        'sc_ya': np.array(part_coeff_data['pc_sc'][:, :101]),
        'sc_oa': np.array(part_coeff_data['pc_sc'][:, 101:])
    }

    return matrices, mod_deg_zscore, part_coeff, ages

def matrix_to_graph(matrix):
    """Convert a matrix to a NetworkX graph."""
    graph = nx.from_numpy_array(matrix)
    return graph

def calculate_sc_features(matrix, mod_deg_zscore, part_coeff, mod_deg_zscore_fc=None, part_coeff_fc=None):
    """
    Calculate node features from a structural connectivity matrix.

    Parameters:
    - matrix: Structural connectivity matrix.
    - mod_deg_zscore: Modular degree z-score for structural connectivity.
    - part_coeff: Participation coefficient for structural connectivity.
    - mod_deg_zscore_fc: Modular degree z-score for functional connectivity (optional).
    - part_coeff_fc: Participation coefficient for functional connectivity (optional).

    Returns:
    - all_subjects_features: List of tensors containing node features for all subjects.
    - feature_names: Names of the features calculated.
    """
    num_subjects = matrix.shape[2]
    all_subjects_features = []
    feature_names = ['mean', 'std', 'skewness', 'kurtosis', 'mod_deg_zscore', 'part_coeff']

    if mod_deg_zscore_fc is not None and part_coeff_fc is not None:
        feature_names += ['mod_deg_zscore_fc', 'part_coeff_fc']

    for s in range(num_subjects):
        node_features = []
        for i in range(matrix.shape[0]):
            connections = matrix[i, :, s].flatten()
            stats = [
                connections.mean(),
                connections.std(),
                scipy.stats.skew(connections),
                scipy.stats.kurtosis(connections)
            ]
            mod_part = [mod_deg_zscore[i, s], part_coeff[i, s]]
            sc_features = stats + mod_part

            if mod_deg_zscore_fc is not None and part_coeff_fc is not None:
                fc_features = [mod_deg_zscore_fc[i, s], part_coeff_fc[i, s]]
                combined_features = sc_features + fc_features
                node_features.append(combined_features)
            else:
                node_features.append(sc_features)

        all_subjects_features.append(torch.tensor(node_features, dtype=torch.float32))

    return all_subjects_features, feature_names

def create_graphs_with_features(matrix, feature_tensor=None, feature_type='random'):
    """
    Combine a connectivity matrix and a feature tensor into a graph.

    Parameters:
    - matrix: Connectivity matrix.
    - feature_tensor: List of feature tensors. If None, feature_type is used.
    - feature_type: Type of tensor to use if feature_tensor is None ('random' or 'identity').

    Returns:
    - graph_list: List of PyTorch Geometric Data objects.
    """
    tensor_matrix = torch.tensor(matrix, dtype=torch.float)
    graph_list = []
    num_subjects = tensor_matrix.shape[2]
    num_nodes = tensor_matrix.shape[0]

    # Generate feature tensor if not provided
    if feature_tensor is None:
        if feature_type == 'random':
            feature_tensor = [torch.rand((num_nodes, 4), dtype=torch.float32) for _ in range(num_subjects)]
        elif feature_type == 'identity':
            feature_tensor = [torch.eye(num_nodes, dtype=torch.float32) for _ in range(num_subjects)]
        else:
            raise ValueError("Invalid feature_type. Choose 'random' or 'identity'.")

    # Create graphs for each subject
    for i in range(num_subjects):
        edges = []
        edge_weights = []
        for j in range(num_nodes):
            for k in range(j+1, num_nodes):
                weight = tensor_matrix[j, k, i]
                if weight != 0:  # Remove null edges
                    edges.append([j, k])
                    edges.append([k, j])
                    edge_weights.append(weight)
                    edge_weights.append(weight)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        data = Data(x=feature_tensor[i], edge_index=edge_index, edge_attr=edge_weights)
        graph_list.append(data)

    return graph_list

def normalize_graph_features(graphs):
    """
    Normalize the node features of each graph individually.
    """
    normalized_graphs = []
    for graph in graphs:
        mean = graph.x.mean(dim=0)
        std = graph.x.std(dim=0) + 1e-8  # To avoid division by zero
        normalized_x = (graph.x - mean) / std
        normalized_graph = copy.deepcopy(graph)
        normalized_graph.x = normalized_x
        normalized_graphs.append(normalized_graph)
    return normalized_graphs

def scale_graph_edge_weights(graphs):
    """
    Scale the edge weights of each graph individually to [0, 1].
    """
    scaled_graphs = []
    for graph in graphs:
        edge_weights = graph.edge_attr
        min_weight = edge_weights.min()
        max_weight = edge_weights.max()
        scaled_weights = (edge_weights - min_weight) / (max_weight - min_weight + 1e-8)
        scaled_graph = copy.deepcopy(graph)
        scaled_graph.edge_attr = scaled_weights
        scaled_graphs.append(scaled_graph)
    return scaled_graphs

def prepare_datasets(base_dir):
    """
    Prepare datasets by loading data, calculating features, and creating graphs.

    Returns:
    - graphs_sc: List of graphs for structural connectivity.
    - labels_sc: Corresponding labels for the graphs.
    - graphs_sc_combined: List of graphs with combined features.
    - labels_sc_combined: Corresponding labels for the combined graphs.
    - feature_names: Names of the features calculated.
    """
    # Load data
    matrices, mod_deg_zscore, part_coeff, ages = load_all_data(base_dir)

    # Calculate node features for structural connectivity
    features_sc_ya = calculate_sc_features(
        matrices['sc_ya'],
        mod_deg_zscore['sc_ya'],
        part_coeff['sc_ya']
    )
    features_sc_oa = calculate_sc_features(
        matrices['sc_oa'],
        mod_deg_zscore['sc_oa'],
        part_coeff['sc_oa']
    )

    # Calculate combined features (including functional connectivity features)
    features_combined_ya = calculate_sc_features(
        matrices['sc_ya'],
        mod_deg_zscore['sc_ya'],
        part_coeff['sc_ya'],
        mod_deg_zscore_fc=mod_deg_zscore['fc_ya'],
        part_coeff_fc=part_coeff['fc_ya']
    )
    features_combined_oa = calculate_sc_features(
        matrices['sc_oa'],
        mod_deg_zscore['sc_oa'],
        part_coeff['sc_oa'],
        mod_deg_zscore_fc=mod_deg_zscore['fc_oa'],
        part_coeff_fc=part_coeff['fc_oa']
    )

    # Create graphs with features
    graphs_with_features = {
        'sc_ya': create_graphs_with_features(
            matrices['sc_ya'], feature_tensor=features_sc_ya[0]
        ),
        'sc_oa': create_graphs_with_features(
            matrices['sc_oa'], feature_tensor=features_sc_oa[0]
        ),
        'sc_combined_ya': create_graphs_with_features(
            matrices['sc_ya'], feature_tensor=features_combined_ya[0]
        ),
        'sc_combined_oa': create_graphs_with_features(
            matrices['sc_oa'], feature_tensor=features_combined_oa[0]
        ),
    }

    # Combine young and old adult graphs and labels
    graphs_sc = graphs_with_features['sc_ya'] + graphs_with_features['sc_oa']
    labels_sc = [0] * len(graphs_with_features['sc_ya']) + [1] * len(graphs_with_features['sc_oa'])

    graphs_sc_combined = graphs_with_features['sc_combined_ya'] + graphs_with_features['sc_combined_oa']
    labels_sc_combined = [0] * len(graphs_with_features['sc_combined_ya']) + [1] * len(graphs_with_features['sc_combined_oa'])

    # Feature names
    feature_names = features_sc_ya[1]

    # Normalize and scale graphs before splitting
    graphs_sc = normalize_graph_features(graphs_sc)
    graphs_sc = scale_graph_edge_weights(graphs_sc)

    graphs_sc_combined = normalize_graph_features(graphs_sc_combined)
    graphs_sc_combined = scale_graph_edge_weights(graphs_sc_combined)

    return graphs_sc, labels_sc, graphs_sc_combined, labels_sc_combined, feature_names
