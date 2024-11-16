from sklearn.model_selection import train_test_split
from dgl.dataloading import GraphDataLoader
import torch.optim as optim
from preprocessing import normalize_features, scale_edge_weights, GraphDataset, collate_fn
import random
import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional: Set environment variable for Python hash seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

    # Seed generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(config.random_state)

    # Data loaders
    train_loader = GraphDataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker
    )
    val_loader = GraphDataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    test_loader = GraphDataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )

    return train_loader, val_loader, test_loader