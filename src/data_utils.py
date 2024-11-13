from sklearn.model_selection import train_test_split
from dgl.dataloading import GraphDataLoader
import torch.optim as optim
from preprocessing import normalize_features, scale_edge_weights, GraphDataset, collate_fn

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def build_dataloaders(graphs, labels, config):
    # Split the data
    train_graphs, temp_graphs, train_labels, temp_labels = train_test_split(
        graphs, labels, test_size=config.test_size, random_state=config.random_state, stratify=labels
    )

    val_graphs, test_graphs, val_labels, test_labels = train_test_split(
        temp_graphs, temp_labels, test_size=0.5, random_state=config.random_state, stratify=temp_labels
    )

    # Preprocess graphs
    normalized_train_graphs, normalized_val_graphs, normalized_test_graphs = normalize_features(
        train_graphs, val_graphs, test_graphs
    )

    scaled_train_graphs = scale_edge_weights(normalized_train_graphs)
    scaled_val_graphs = scale_edge_weights(normalized_val_graphs)
    scaled_test_graphs = scale_edge_weights(normalized_test_graphs)
        
    # Create Datasets
    train_dataset = GraphDataset(scaled_train_graphs, train_labels)
    val_dataset = GraphDataset(scaled_val_graphs, val_labels)
    test_dataset = GraphDataset(scaled_test_graphs, test_labels)

    # Data loaders
    train_loader = GraphDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = GraphDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = GraphDataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader