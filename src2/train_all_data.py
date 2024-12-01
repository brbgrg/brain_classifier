import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_test import train, validate
from prepare_datasets import compute_feature_means_stds, compute_edge_attr_means_stds, normalize_graph_features, normalize_graph_edge_weights
from build_dataloader import build_dataloader, set_seed, GraphDataset
from model import GAT
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model_on_all_data(
    graphs,
    labels,
    dataset_name,
    best_params,
    model_class,
    max_epochs,
    batch_size,
    device
):
    """
    Trains the model on all the provided graphs and labels for a specified number of epochs,
    saves the best model, and plots training curves.

    Args:
        graphs (list): List of graph data objects.
        labels (list): List of labels corresponding to the graphs.
        dataset_name (str): Name of the dataset.
        best_params (dict): Dictionary of best hyperparameters.
        max_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for data loader.
        device (torch.device): Device to run the model on.
        test_graphs (list, optional): List of test graph data objects.
        test_labels (list, optional): List of labels corresponding to the test graphs.
    """
    # Compute means and stds on the training data
    mean_x, std_x = compute_feature_means_stds(graphs)
    mean_edge_attr, std_edge_attr = compute_edge_attr_means_stds(graphs)

    # Normalize the training data
    graphs = normalize_graph_features(graphs, mean_x, std_x)
    graphs = normalize_graph_edge_weights(graphs, mean_edge_attr, std_edge_attr)

    # Build dataset and data loader
    dataset = GraphDataset(graphs, labels)
    data_loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)


    # Get the number of node features
    num_node_features = graphs[0].x.size(1)

    # Initialize the model with best hyperparameters
    model = model_class(
        in_channels=num_node_features,
        out_channels=best_params['out_channels'],
        num_heads=best_params.get('num_heads', 1),
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    # Initialize lists to store metrics
    loss_history = []
    accuracy_history = []
    f1_history = []

    # Initialize best_loss to track the best model
    best_loss = float('inf')

    for epoch in range(1, max_epochs + 1):
        # Training step
        loss, accuracy, f1 = train(data_loader, model, criterion, optimizer, device)

        # Append metrics to history
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        f1_history.append(f1)

        print(f"Epoch {epoch}/{max_epochs}: Loss: {loss:.4f}, F1: {f1:.4f}")

        # Save the best model
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'in_channels': num_node_features,
                    'out_channels': best_params['out_channels'],
                    'num_heads': best_params['num_heads'],
                    'batch_size': batch_size,
                    'learning_rate': best_params['learning_rate'],
                    'weight_decay': best_params['weight_decay'],
                }
            }, f'best_model_{dataset_name}.pth')

    # Plot training curves
    epochs_range = range(1, len(loss_history) + 1)
    loss_series = pd.Series(loss_history)
    accuracy_series = pd.Series(accuracy_history)
    f1_series = pd.Series(f1_history)
    window_size = 3

    # Calculate exponential moving averages
    loss_ema = loss_series.ewm(span=window_size, adjust=False).mean()
    accuracy_ema = accuracy_series.ewm(span=window_size, adjust=False).mean()
    f1_ema = f1_series.ewm(span=window_size, adjust=False).mean()

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Loss
    axes[0].plot(epochs_range, loss_history, label='Loss', marker='o')
    axes[0].plot(epochs_range, loss_ema, label='Loss EMA', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracy
    axes[1].plot(epochs_range, accuracy_history, label='Accuracy', marker='o')
    axes[1].plot(epochs_range, accuracy_ema, label='Accuracy EMA', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True)

    # Plot F1 Score
    axes[2].plot(epochs_range, f1_history, label='F1 Score', marker='o')
    axes[2].plot(epochs_range, f1_ema, label='F1 EMA', linestyle='--')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Training F1 Score over Epochs')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Clean up
    del model
    del optimizer
    del criterion
    torch.cuda.empty_cache()
