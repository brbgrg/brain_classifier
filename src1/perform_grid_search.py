# grid_search.py

import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from train_test import train, validate
from data_preprocessing import GraphDataset, collate_fn
from dgl.dataloading import GraphDataLoader
import torch.nn as nn
import torch.optim as optim

def perform_grid_search(train_graphs, train_labels, num_splits, param_grid, batch_size, model_class, device):
    # Get the number of node features from the data
    num_node_features = train_graphs[0].x.size(1)  # Assuming graphs are PyG Data objects

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Initialize list to store all results
    all_results = []

    # Remove 'in_channels' from param_grid since it's determined by data
    param_grid = [{k: v for k, v in params.items() if k != 'in_channels'} for params in ParameterGrid(param_grid)]

    # Loop over each parameter combination
    for params in param_grid:
        num_heads = params['num_heads']
        out_channels = params['out_channels']
        num_epochs = params['num_epochs']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']

        # Store per-fold results
        fold_results = []

        for fold, (train_index, val_index) in enumerate(kf.split(train_graphs, train_labels)):
            # Split the dataset into training and validation sets for this fold
            fold_train_data = [train_graphs[i] for i in train_index]
            fold_val_data = [train_graphs[i] for i in val_index]

            fold_train_labels = [train_labels[i] for i in train_index]
            fold_val_labels = [train_labels[i] for i in val_index]

            # Create PyTorch datasets and data loaders
            fold_train_dataset = GraphDataset(fold_train_data, fold_train_labels)
            fold_val_dataset = GraphDataset(fold_val_data, fold_val_labels)

            # Create data loaders
            fold_train_loader = GraphDataLoader(
                fold_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            fold_val_loader = GraphDataLoader(
                fold_val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            # Initialize the model, criterion, and optimizer
            model = model_class(
                in_channels=num_node_features,
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
                train_loss, train_accuracy, train_f1 = train(fold_train_loader, model, criterion, optimizer, device)
                val_loss, val_accuracy, val_f1 = validate(fold_val_loader, model, criterion, device)

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
            del model
            del optimizer
            del criterion
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
