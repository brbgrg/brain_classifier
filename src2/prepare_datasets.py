# data_ingestion.py

import os
import numpy as np
import scipy.io
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.stats
import copy
from sklearn.model_selection import train_test_split

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
    feature_names = ['mean', 'std', 'skewness', 'kurtosis', 'mod_deg_zscore_sc', 'part_coeff_sc']

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
            mod_part_sc = [mod_deg_zscore[i, s], part_coeff[i, s]]
            sc_features = stats + mod_part_sc

            if mod_deg_zscore_fc is not None and part_coeff_fc is not None:
                mod_part_fc = [mod_deg_zscore_fc[i, s], part_coeff_fc[i, s]]
                combined_features = sc_features + mod_part_fc
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

def compute_feature_means_stds(graphs):
    """
    Compute the mean and std of node features over a list of graphs.
    Returns mean and std tensors.
    """
    # Stack all node features
    all_features = torch.cat([graph.x for graph in graphs], dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    return mean, std

def compute_edge_attr_means_stds(graphs):
    """
    Compute the mean and std of edge attributes over a list of graphs.
    Returns mean and std tensors.
    """
    # Stack all edge attributes
    all_edge_attrs = torch.cat([graph.edge_attr for graph in graphs], dim=0)
    mean = all_edge_attrs.mean(dim=0)
    std = all_edge_attrs.std(dim=0)
    return mean, std


def normalize_graph_features(graphs, mean, std):
    """
    Normalize the node features of graphs using the provided mean and std.
    Returns a list of normalized graphs.
    """
    normalized_graphs = []
    for graph in graphs:
        normalized_x = (graph.x - mean) / (std + 1e-8)
        normalized_graph = copy.deepcopy(graph)
        normalized_graph.x = normalized_x
        normalized_graphs.append(normalized_graph)
    return normalized_graphs

def normalize_graph_edge_weights(graphs, mean, std):
    """
    Normalize the edge attributes of graphs using the provided mean and std.
    Returns a list of normalized graphs.
    """
    normalized_graphs = []
    for graph in graphs:
        normalized_edge_attr = (graph.edge_attr - mean) / (std + 1e-8)
        normalized_graph = copy.deepcopy(graph)
        normalized_graph.edge_attr = normalized_edge_attr
        normalized_graphs.append(normalized_graph)
    return normalized_graphs


def prepare_datasets(base_dir, test_size=0.15, random_state=42):
    """
    Prepare datasets by loading data, calculating features, creating graphs,
    and splitting into training and testing sets.

    Returns:
    - datasets: A dictionary containing the following keys:
        - 'train_graphs_sc': Training graphs with SC features only.
        - 'train_labels_sc': Corresponding labels for the SC training graphs.
        - 'test_graphs_sc': Testing graphs with SC features only.
        - 'test_labels_sc': Corresponding labels for the SC testing graphs.
        - 'train_graphs_sc_fc': Training graphs with combined SC and FC features.
        - 'train_labels_sc_fc': Corresponding labels for the combined training graphs.
        - 'test_graphs_sc_fc': Testing graphs with combined SC and FC features.
        - 'test_labels_sc_fc': Corresponding labels for the combined testing graphs.
    - feature_names_sc: Names of the SC features calculated.
    - feature_names_combined: Names of the combined features calculated.
    """
    # Load data
    matrices, mod_deg_zscore, part_coeff, ages = load_all_data(base_dir)

    # Calculate node features for structural connectivity
    features_sc_ya, feature_names_sc = calculate_sc_features(
        matrices['sc_ya'],
        mod_deg_zscore['sc_ya'],
        part_coeff['sc_ya']
    )
    features_sc_oa, _ = calculate_sc_features(
        matrices['sc_oa'],
        mod_deg_zscore['sc_oa'],
        part_coeff['sc_oa']
    )

    # Calculate combined features (including functional connectivity features)
    features_combined_ya, feature_names_combined = calculate_sc_features(
        matrices['sc_ya'],
        mod_deg_zscore['sc_ya'],
        part_coeff['sc_ya'],
        mod_deg_zscore_fc=mod_deg_zscore['fc_ya'],
        part_coeff_fc=part_coeff['fc_ya']
    )
    features_combined_oa, _ = calculate_sc_features(
        matrices['sc_oa'],
        mod_deg_zscore['sc_oa'],
        part_coeff['sc_oa'],
        mod_deg_zscore_fc=mod_deg_zscore['fc_oa'],
        part_coeff_fc=part_coeff['fc_oa']
    )

    # Create graphs with features
    graphs_with_features = {
        'sc_ya': create_graphs_with_features(
            matrices['sc_ya'], feature_tensor=features_sc_ya
        ),
        'sc_oa': create_graphs_with_features(
            matrices['sc_oa'], feature_tensor=features_sc_oa
        ),
        'sc_combined_ya': create_graphs_with_features(
            matrices['sc_ya'], feature_tensor=features_combined_ya
        ),
        'sc_combined_oa': create_graphs_with_features(
            matrices['sc_oa'], feature_tensor=features_combined_oa
        ),
    }

    # Combine young and old adult graphs and labels for SC features
    graphs_sc = graphs_with_features['sc_ya'] + graphs_with_features['sc_oa']
    labels_sc = [0] * len(graphs_with_features['sc_ya']) + [1] * len(graphs_with_features['sc_oa'])

    # Combine young and old adult graphs and labels for combined features
    graphs_sc_fc = graphs_with_features['sc_combined_ya'] + graphs_with_features['sc_combined_oa']
    labels_sc_fc = [0] * len(graphs_with_features['sc_combined_ya']) + [1] * len(graphs_with_features['sc_combined_oa'])

    # Split into training and testing sets for SC features
    train_graphs_sc, test_graphs_sc, train_labels_sc, test_labels_sc = train_test_split(
        graphs_sc, labels_sc, test_size=test_size, random_state=random_state, stratify=labels_sc
    )

    # Split into training and testing sets for combined features
    train_graphs_sc_fc, test_graphs_sc_fc, train_labels_sc_fc, test_labels_sc_fc = train_test_split(
        graphs_sc_fc, labels_sc_fc, test_size=test_size, random_state=random_state, stratify=labels_sc_fc
    )

    datasets = {
        'train_graphs_sc': train_graphs_sc,
        'train_labels_sc': train_labels_sc,
        'test_graphs_sc': test_graphs_sc,
        'test_labels_sc': test_labels_sc,
        'train_graphs_sc_fc': train_graphs_sc_fc,
        'train_labels_sc_fc': train_labels_sc_fc,
        'test_graphs_sc_fc': test_graphs_sc_fc,
        'test_labels_sc_fc': test_labels_sc_fc
    }

    return datasets, feature_names_sc, feature_names_combined
