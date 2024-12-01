import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_test import train, validate
from build_dataloader import build_dataloader, set_seed, GraphDataset
from prepare_datasets import compute_feature_means_stds, compute_edge_attr_means_stds, normalize_graph_features, normalize_graph_edge_weights
from model import GAT
import pandas as pd
import numpy as np  
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from datetime import datetime

def perform_cv_early_stopping(
    train_graphs,
    train_labels,
    num_splits,
    param_grid,
    model_class,
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
    metrics_per_combination = []

    # Generate all combinations of hyperparameters
    param_grid = list(ParameterGrid(param_grid))

    # Loop over each parameter combination
    for params in param_grid:
        num_heads = params.get('num_heads', 1)  # Default to 1 if not provided
        out_channels = params['out_channels']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']

        print(f"Evaluating hyperparameters: {params}")

        # Store per-fold per-epoch metrics
        fold_metrics_list = []

        # Store per-fold final metrics
        fold_results = []

        for fold, (train_index, val_index) in enumerate(kf.split(train_graphs, train_labels)):
            print(f"Starting fold {fold + 1}/{num_splits}")
            # Split the dataset into training and validation sets for this fold
            fold_train_data = [train_graphs[i] for i in train_index]
            fold_val_data = [train_graphs[i] for i in val_index]

            fold_train_labels = [train_labels[i] for i in train_index]
            fold_val_labels = [train_labels[i] for i in val_index]

            # Compute means and stds on fold_train_data
            mean_x, std_x = compute_feature_means_stds(fold_train_data)
            mean_edge_attr, std_edge_attr = compute_edge_attr_means_stds(fold_train_data)

            # Normalize the training and validation data
            fold_train_data = normalize_graph_features(fold_train_data, mean_x, std_x)
            fold_train_data = normalize_graph_edge_weights(fold_train_data, mean_edge_attr, std_edge_attr)

            fold_val_data = normalize_graph_features(fold_val_data, mean_x, std_x)
            fold_val_data = normalize_graph_edge_weights(fold_val_data, mean_edge_attr, std_edge_attr)

            # Create PyTorch datasets and data loaders
            fold_train_dataset = GraphDataset(fold_train_data, fold_train_labels)
            fold_val_dataset = GraphDataset(fold_val_data, fold_val_labels)

            # Create data loaders
            fold_train_loader = build_dataloader(fold_train_dataset, batch_size=batch_size, shuffle=True)
            fold_val_loader = build_dataloader(fold_val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize the model, criterion, and optimizer
            model = model_class(
                in_channels=num_node_features,
                out_channels=out_channels,
                num_heads=num_heads,
                num_classes=2,
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Early stopping parameters
            best_val_loss_fold = float('inf')
            best_epoch = 0
            epochs_without_improvement = 0
            delta = 1e-4  # Minimum change to qualify as an improvement

            # Initialize metrics for this fold
            metrics = {
                'train_loss': [],
                'val_loss': [],
                'train_accuracy': [],
                'val_accuracy': [],
                'train_f1': [],
                'val_f1': [],
                'epochs': []
            }

            for epoch in range(1, max_epochs + 1):
                train_loss, train_accuracy, train_f1 = train(fold_train_loader, model, criterion, optimizer, device)
                val_loss, val_accuracy, val_f1 = validate(fold_val_loader, model, criterion, device)

                # Add metrics to lists
                metrics['train_loss'].append(train_loss)
                metrics['val_loss'].append(val_loss)
                metrics['train_accuracy'].append(train_accuracy)
                metrics['val_accuracy'].append(val_accuracy)
                metrics['train_f1'].append(train_f1)
                metrics['val_f1'].append(val_f1)
                metrics['epochs'].append(epoch)

                # Early stopping check based on loss
                if best_val_loss_fold - val_loss > delta:
                    best_val_loss_fold = val_loss
                    epochs_without_improvement = 0
                    # Record best metrics
                    best_epoch = epoch
                    best_train_loss = train_loss
                    best_train_accuracy = train_accuracy
                    best_train_f1 = train_f1
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy
                    best_val_f1 = val_f1
                    # Save the model state if needed
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch} in fold {fold + 1}")
                    break

            # Append fold metrics
            fold_metrics_list.append(metrics)

            # Collect best metrics for this fold
            fold_result = {
                'fold': fold + 1,
                'num_heads': num_heads,
                'out_channels': out_channels,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'best_epoch': best_epoch,
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

        # Identify the best fold based on validation F1 score
        best_fold_idx = np.argmax([fr['val_f1'] for fr in fold_results])
        best_fold_metrics = fold_metrics_list[best_fold_idx]

        # Calculate average metrics across folds
        avg_metrics = {}
        for key in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'train_f1', 'val_f1']:
            # Pad shorter lists with NaNs
            max_len = max(len(fm[key]) for fm in fold_metrics_list)
            padded = np.array([np.pad(fm[key], (0, max_len - len(fm[key])), constant_values=np.nan) for fm in fold_metrics_list])
            avg_metrics[key] = np.nanmean(padded, axis=0)
            # Compute standard deviation
            avg_metrics[f'{key}_std'] = np.nanstd(padded, axis=0)
        avg_metrics['epochs'] = list(range(1, len(avg_metrics['train_loss']) + 1))

        # Store metrics for plotting
        metrics_per_combination.append({
            'hyperparams': {
                'num_heads': num_heads,
                'out_channels': out_channels,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            },
            'avg_metrics': avg_metrics,
            'best_fold_metrics': best_fold_metrics,
            'fold_metrics_list': fold_metrics_list,  # For individual fold plots
            'best_epoch': np.mean([fr['best_epoch'] for fr in fold_results])  # Average best epoch across folds
        })
            
    # Now, plot all hyperparameter combinations in a single figure
    n_combinations = len(metrics_per_combination)
    fig, axes = plt.subplots(n_combinations, 3, figsize=(18, 6 * n_combinations))

    if n_combinations == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D

    for idx, data in enumerate(metrics_per_combination):
        avg_metrics = data['avg_metrics']
        epochs = avg_metrics['epochs']
        hyperparams = data['hyperparams']
        best_epoch = data['best_epoch']
        hyperparam_text = (
            f"num_heads={hyperparams['num_heads']}, out_channels={hyperparams['out_channels']}, "
            f"lr={hyperparams['learning_rate']}, wd={hyperparams['weight_decay']}, best_epoch={int(best_epoch)}"
        )

        # Compute moving averages
        window_size = 10  # Adjust the window size as needed
        for key in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'train_f1', 'val_f1']:
            # Moving average of the average
            avg_metrics[f'{key}_ma'] = pd.Series(avg_metrics[key]).rolling(window=window_size, min_periods=1).mean()
            # Moving average of the std deviation
            avg_metrics[f'{key}_std_ma'] = pd.Series(avg_metrics[f'{key}_std']).rolling(window=window_size, min_periods=1).mean()
        epochs_ma = avg_metrics['epochs']

        # Plot Loss
        ax = axes[idx, 0]

        # Plot moving average of the average curves with moving average of standard deviation
        ax.plot(epochs_ma, avg_metrics['train_loss_ma'], label='Moving Avg Train Loss', color='blue')
        ax.fill_between(
            epochs_ma,
            avg_metrics['train_loss_ma'] - avg_metrics['train_loss_std_ma'],
            avg_metrics['train_loss_ma'] + avg_metrics['train_loss_std_ma'],
            color='blue',
            alpha=0.2
        )
        ax.plot(epochs_ma, avg_metrics['val_loss_ma'], label='Moving Avg Validation Loss', color='red')
        ax.fill_between(
            epochs_ma,
            avg_metrics['val_loss_ma'] - avg_metrics['val_loss_std_ma'],
            avg_metrics['val_loss_ma'] + avg_metrics['val_loss_std_ma'],
            color='red',
            alpha=0.2
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss over Epochs')
        ax.text(0.5, -0.12, hyperparam_text, transform=ax.transAxes, ha='center', fontsize=10)
        ax.legend()
        ax.grid(True)

        # Plot Accuracy
        ax = axes[idx, 1]
        ax.plot(epochs_ma, avg_metrics['train_accuracy_ma'], label='Moving Avg Train Accuracy', color='blue')
        ax.fill_between(
            epochs_ma,
            avg_metrics['train_accuracy_ma'] - avg_metrics['train_accuracy_std_ma'],
            avg_metrics['train_accuracy_ma'] + avg_metrics['train_accuracy_std_ma'],
            color='blue',
            alpha=0.2
        )
        ax.plot(epochs_ma, avg_metrics['val_accuracy_ma'], label='Moving Avg Validation Accuracy', color='red')
        ax.fill_between(
            epochs_ma,
            avg_metrics['val_accuracy_ma'] - avg_metrics['val_accuracy_std_ma'],
            avg_metrics['val_accuracy_ma'] + avg_metrics['val_accuracy_std_ma'],
            color='red',
            alpha=0.2
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy over Epochs')
        ax.legend()
        ax.grid(True)

        # Plot F1 Score
        ax = axes[idx, 2]
        ax.plot(epochs_ma, avg_metrics['train_f1_ma'], label='Moving Avg Train F1 Score', color='blue')
        ax.fill_between(
            epochs_ma,
            avg_metrics['train_f1_ma'] - avg_metrics['train_f1_std_ma'],
            avg_metrics['train_f1_ma'] + avg_metrics['train_f1_std_ma'],
            color='blue',
            alpha=0.2
        )
        ax.plot(epochs_ma, avg_metrics['val_f1_ma'], label='Moving Avg Validation F1 Score', color='red')
        ax.fill_between(
            epochs_ma,
            avg_metrics['val_f1_ma'] - avg_metrics['val_f1_std_ma'],
            avg_metrics['val_f1_ma'] + avg_metrics['val_f1_std_ma'],
            color='red',
            alpha=0.2
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score over Epochs')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig(f'metrics_{timestamp}.png')
    plt.close(fig)

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
    best_epochs_trained = int(best_row['best_epoch'])  # Average best epoch across folds
    with open(f'best_hyperparameters_{timestamp}.txt', 'w') as f:
        f.write(f"Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"Average number of epochs trained: {best_epochs_trained}\n")
        f.write(f"Best Validation F1 Score: {best_val_f1}\n")
        f.write(f"Best Validation Accuracy: {best_val_accuracy}\n")

    # Return the best parameters and the results DataFrame
    return best_params, best_val_f1, best_val_accuracy, best_epochs_trained, results_df
