# sweep.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import copy
import dgl
from dgl.dataloading import GraphDataLoader
import wandb
from ingestion import prepare_datasets
from sklearn.metrics import accuracy_score, f1_score
from model import GAT  # Import the model from model.py
from preprocessing import normalize_features, scale_edge_weights, GraphDataset, collate_fn
from model import GAT
from train_and_test import train, validate

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

def run_sweep(graphs, labels, sweep_config, dataset_name):
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='graph-classification-' + dataset_name)
    # Function to execute a single run
    def train_sweep():
        wandb.init()
        config = wandb.config

        # Build data loaders
        train_loader, val_loader, test_loader = build_dataloaders(graphs, labels, config)

        # Model, criterion, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GAT(
            in_channels=config.in_channels, 
            out_channels=config.out_channels, 
            num_heads=config.num_heads, 
            num_classes=config.num_classes
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

        # Watch the model with wandb
        wandb.watch(model, log="all", log_freq=10)

        # Train the model
        best_val_accuracy = 0.0
        best_model_state = None
        for epoch in range(config.num_epochs):
            train_loss, train_accuracy, train_f1 = train(train_loader, model, criterion, optimizer, device)
            val_loss, val_accuracy, val_f1 = validate(val_loader, model, criterion, device)

            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
            })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save({
                    'model_state_dict': model.state_dict(), 
                    'random_state': config.random_state #or whole config
                }, f'best_model_{dataset_name}.pth')
                wandb.save(f'best_model_{dataset_name}.pth')

        # Finish the wandb run
        wandb.finish()

    wandb.agent(sweep_id, function=train_sweep)
