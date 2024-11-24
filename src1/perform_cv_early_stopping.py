import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_test import train, validate
from build_dataloader import build_dataloader, set_seed, GraphDataset
from model import GAT
import pandas as pd
import numpy as np  
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from datetime import datetime


def perform_cv_early_stopping(
    train_graphs,
    train_labels,
    num_splits,
    param_grid,
    batch_size,
    device,
    patience,
    max_epochs
):
    set_seed(42)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    num_node_features = train_graphs[0].x.size(1)  # Assuming graphs are PyG Data objects
    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Initialize lists to store all results and average metrics
    all_results = []
    avg_metrics_per_combination = []

    # Generate all combinations of hyperparameters
    param_grid = list(ParameterGrid(param_grid))

    # Loop over each parameter combination
    for params in param_grid:
        num_heads = params['num_heads']
        out_channels = params['out_channels']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']

        # Store per-fold per-epoch metrics
        fold_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'epochs': []
        }

        # Store per-fold number of epochs
        fold_num_epochs = []

        # Store per-fold final metrics
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
            fold_train_loader = build_dataloader(fold_train_dataset, batch_size=batch_size, shuffle=True)
            fold_val_loader = build_dataloader(fold_val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize the model, criterion, and optimizer
            model = GAT(
                in_channels=num_node_features,
                out_channels=out_channels,
                num_heads=num_heads,
                num_classes=2,
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Early stopping parameters
            best_val_loss = float('inf')
            best_epoch = 0
            epochs_without_improvement = 0
            delta = 1e-4  # Minimum change to qualify as an improvement


            # Initialize best metrics
            best_train_loss = None
            best_train_accuracy = None
            best_train_f1 = None
            best_val_accuracy = None
            best_val_f1 = None
  
            # Initialize metrics for this fold
            train_loss_history = []
            val_loss_history = []
            train_accuracy_history = []
            val_accuracy_history = []
            train_f1_history = []
            val_f1_history = []

            for epoch in range(1, max_epochs + 1):
                train_loss, train_accuracy, train_f1 = train(fold_train_loader, model, criterion, optimizer, device)
                val_loss, val_accuracy, val_f1 = validate(fold_val_loader, model, criterion, device)

                # Add metrics to lists
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                train_accuracy_history.append(train_accuracy)
                val_accuracy_history.append(val_accuracy)
                train_f1_history.append(train_f1)
                val_f1_history.append(val_f1)

                # Early stopping check
                if best_val_loss - val_loss > delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Record best metrics
                    best_epoch = epoch
                    best_train_loss = train_loss
                    best_train_accuracy = train_accuracy
                    best_train_f1 = train_f1
                    best_val_accuracy = val_accuracy
                    best_val_f1 = val_f1
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch} in fold {fold+1}")
                    break

            # Store per-fold per-epoch metrics
            fold_metrics['train_loss'].append(train_loss_history)
            fold_metrics['val_loss'].append(val_loss_history)
            fold_metrics['train_accuracy'].append(train_accuracy_history)
            fold_metrics['val_accuracy'].append(val_accuracy_history)
            fold_metrics['train_f1'].append(train_f1_history)
            fold_metrics['val_f1'].append(val_f1_history)
            fold_metrics['epochs'].append(len(train_loss_history))

            # Store number of epochs in this fold
            fold_num_epochs.append(len(train_loss_history))

            # Collect best metrics for this fold
            fold_result = {
                'fold': fold + 1,
                'num_heads': num_heads,
                'out_channels': out_channels,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'epochs_trained': best_epoch,
                'train_loss': best_train_loss,
                'val_loss': best_val_loss,
                'train_accuracy': best_train_accuracy,
                'val_accuracy': best_val_accuracy,
                'train_f1': best_train_f1,
                'val_f1': best_val_f1
            }
            fold_results.append(fold_result)


            # Clean up to free memory
            del model
            del optimizer
            del criterion
            torch.cuda.empty_cache()

        # Append fold results to the overall results
        all_results.extend(fold_results)

        # Now, average the per-epoch metrics over folds

        # First, find the maximum number of epochs across folds
        max_num_epochs = max(fold_num_epochs)

        # Initialize lists to store averaged per-epoch metrics
        avg_train_loss = []
        avg_val_loss = []
        avg_train_accuracy = []
        avg_val_accuracy = []
        avg_train_f1 = []
        avg_val_f1 = []

        # For each epoch up to max_num_epochs
        for epoch_idx in range(max_num_epochs):
            epoch_train_loss = []
            epoch_val_loss = []
            epoch_train_accuracy = []
            epoch_val_accuracy = []
            epoch_train_f1 = []
            epoch_val_f1 = []

            for fold_idx in range(num_splits):
                if epoch_idx < fold_num_epochs[fold_idx]:
                    # Append the metric for this epoch and fold
                    epoch_train_loss.append(fold_metrics['train_loss'][fold_idx][epoch_idx])
                    epoch_val_loss.append(fold_metrics['val_loss'][fold_idx][epoch_idx])
                    epoch_train_accuracy.append(fold_metrics['train_accuracy'][fold_idx][epoch_idx])
                    epoch_val_accuracy.append(fold_metrics['val_accuracy'][fold_idx][epoch_idx])
                    epoch_train_f1.append(fold_metrics['train_f1'][fold_idx][epoch_idx])
                    epoch_val_f1.append(fold_metrics['val_f1'][fold_idx][epoch_idx])

            # Compute average for this epoch across folds
            if epoch_train_loss:
                avg_train_loss.append(np.mean(epoch_train_loss))
                avg_val_loss.append(np.mean(epoch_val_loss))
                avg_train_accuracy.append(np.mean(epoch_train_accuracy))
                avg_val_accuracy.append(np.mean(epoch_val_accuracy))
                avg_train_f1.append(np.mean(epoch_train_f1))
                avg_val_f1.append(np.mean(epoch_val_f1))
            else:
                break

        # Store the averaged metrics along with hyperparameters
        avg_metrics_per_combination.append({
            'hyperparams': {
                'num_heads': num_heads,
                'out_channels': out_channels,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            },
            'epochs': list(range(1, len(avg_train_loss) + 1)),
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
            'avg_train_accuracy': avg_train_accuracy,
            'avg_val_accuracy': avg_val_accuracy,
            'avg_train_f1': avg_train_f1,
            'avg_val_f1': avg_val_f1
        })

    # Now, plot all hyperparameter combinations in a single figure
    n_combinations = len(avg_metrics_per_combination)
    fig, axes = plt.subplots(n_combinations, 3, figsize=(18, 6 * n_combinations))

    if n_combinations == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D

    for idx, metrics in enumerate(avg_metrics_per_combination):
        epochs = metrics['epochs']
        hyperparams = metrics['hyperparams']
        hyperparam_text = f"num_heads={hyperparams['num_heads']}, out_channels={hyperparams['out_channels']}, lr={hyperparams['learning_rate']}, wd={hyperparams['weight_decay']}"

        # Plot Loss
        axes[idx, 0].plot(epochs, metrics['avg_train_loss'], label='Avg Train Loss', marker='o')
        axes[idx, 0].plot(epochs, metrics['avg_val_loss'], label='Avg Validation Loss', marker='o')
        axes[idx, 0].set_xlabel('Epoch')
        axes[idx, 0].set_ylabel('Loss')
        axes[idx, 0].set_title('Average Loss over Epochs')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)

        # Plot Accuracy
        axes[idx, 1].plot(epochs, metrics['avg_train_accuracy'], label='Avg Train Accuracy', marker='o')
        axes[idx, 1].plot(epochs, metrics['avg_val_accuracy'], label='Avg Validation Accuracy', marker='o')
        axes[idx, 1].set_xlabel('Epoch')
        axes[idx, 1].set_ylabel('Accuracy')
        axes[idx, 1].set_title('Average Accuracy over Epochs')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
  
        axes[idx, 2].plot(epochs, metrics['avg_val_f1'], label='Avg Validation F1 Score', marker='o')
        axes[idx, 2].set_xlabel('Epoch')
        axes[idx, 2].set_ylabel('F1 Score')
        axes[idx, 2].set_title('Average F1 Score over Epochs')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True)

        # Add hyperparameter configuration text
    fig.tight_layout()
    fig.savefig(f'metrics_{timestamp}.png')
    plt.close()
    plt.savefig('metrics.png')
    plt.close()

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
    best_epochs_trained = int(best_row['epochs_trained'])
    with open(f'best_hyperparameters_{timestamp}.txt', 'w') as f:
        f.write(f"Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"Number of epochs trained: {best_epochs_trained}\n")
        f.write(f"Best Validation F1 Score: {best_val_f1}\n")
        f.write(f"Best Validation Accuracy: {best_val_accuracy}\n")
        f.write(f"Best Validation F1 Score: {best_val_f1}\n")
        f.write(f"Best Validation Accuracy: {best_val_accuracy}\n")

    # Return the best parameters and the results DataFrame
    return best_params, best_val_f1, best_val_accuracy, best_epochs_trained, results_df

