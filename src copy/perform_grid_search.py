#perform_grid_search.py
# Kfold cross-validation

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from train_and_test import train, validate
from data_utils import GraphDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def perform_grid_search(graphs, labels, num_splits, param_grid, batch_size, model_class):
    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Initialize list to store all results
    all_results = []

    # Loop over each parameter combination
    for params in ParameterGrid(param_grid):
        num_heads = params['num_heads']
        in_channels = params['in_channels']
        out_channels = params['out_channels']
        num_epochs = params['num_epochs']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']  # If weight_decay is in your param_grid

        # Store per-fold results
        fold_results = []

        for fold, (train_index, val_index) in enumerate(kf.split(graphs, labels)):
            # Split the dataset into training and validation sets for this fold
            train_data = [graphs[i] for i in train_index]
            val_data = [graphs[i] for i in val_index]

            train_labels = [labels[i] for i in train_index]
            val_labels = [labels[i] for i in val_index]

            # Create PyTorch datasets and data loaders
            train_dataset = GraphDataset(train_data, train_labels)
            val_dataset = GraphDataset(val_data, val_labels)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize the model, criterion, and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model_class(
                in_channels=in_channels,  # Replace with the appropriate value
                out_channels=out_channels,
                num_heads=num_heads,
                num_classes=2,  
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Initialize metrics for this fold
            train_loss_history = []
            val_loss_history = []
            train_accuracy_history = []
            val_accuracy_history = []
            train_f1_history = []
            val_f1_history = []

            for epoch in range(num_epochs):
                train_loss, train_accuracy, train_f1 = train(train_loader, model, criterion, optimizer, device)
                val_loss, val_accuracy, val_f1 = validate(val_loader, model, criterion, device)

                # Add metrics to lists
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                train_accuracy_history.append(train_accuracy)
                val_accuracy_history.append(val_accuracy)
                train_f1_history.append(train_f1)
                val_f1_history.append(val_f1)

            # Collect final metrics for this fold
            fold_result = {
                'fold': fold + 1,
                'num_heads': num_heads,
                'out_channels': out_channels,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'train_loss': train_loss_history[-1],
                'val_loss': val_loss_history[-1],
                'train_accuracy': train_accuracy_history[-1],
                'val_accuracy': val_accuracy_history[-1],
                'train_f1': train_f1_history[-1],
                'val_f1': val_f1_history[-1]
            }
            fold_results.append(fold_result)

            # Clean up to free memory
            del model_instance
            del optimizer_instance
            del criterion_instance
            torch.cuda.empty_cache()

        # Append fold results to the overall results
        all_results.extend(fold_results)

    # Create a DataFrame from all_results
    results_df = pd.DataFrame(all_results)

    # Compute average metrics across folds for each parameter combination
    hyperparams = ['num_heads', 'out_channels', 'learning_rate', 'weight_decay']
    avg_results_df = results_df.groupby(hyperparams).mean().reset_index()

    # Find the best hyperparameters based on average validation F1 score
    best_row = avg_results_df.loc[avg_results_df['val_f1'].idxmax()]
    best_params = best_row[hyperparams].to_dict()
    best_val_f1 = best_row['val_f1']
    best_val_accuracy = best_row['val_accuracy']

    # Return the best parameters and the results DataFrame
    return best_params, best_val_f1, best_val_accuracy, results_df
