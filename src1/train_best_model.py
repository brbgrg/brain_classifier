import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_test import train, validate
from build_dataloader import build_dataloader, set_seed, GraphDataset
from model import GAT
from sklearn.model_selection import train_test_split
import pandas as pd


def train_model_with_early_stopping(graphs, labels, dataset_name, best_params, patience, max_epochs, batch_size, device):
    """
    Trains the model with the best hyperparameters using early stopping and plots training curves.

    Args:
        graphs (list): List of graph data objects.
        labels (list): List of labels corresponding to the graphs.
        best_params (dict): Dictionary of best hyperparameters.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        max_epochs (int): Maximum number of epochs to train.
        batch_size (int): Batch size for data loaders.
        device (torch.device): Device to run the model on.

    Returns:
        None: The function plots the training and validation curves.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Split the data
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        graphs, labels, test_size=0.17, random_state=42, stratify=labels
    )

    # Build datasets and data loaders
    train_dataset = GraphDataset(train_graphs, train_labels)
    val_dataset = GraphDataset(val_graphs, val_labels)

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get the number of node features from the data
    num_node_features = graphs[0].x.size(1)  # Assuming graphs are DGL or PyG Data objects
    
    # Initialize the model with best hyperparameters
    model = GAT(
        in_channels=num_node_features,
        out_channels=best_params['out_channels'],
        num_heads=best_params['num_heads'],
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    delta = 1e-4  # Minimum change to qualify as an improvement

    
    # Initialize lists to store metrics
    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    train_f1_history = []
    val_f1_history = []
    
    for epoch in range(1, max_epochs + 1):
        # Training step
        train_loss, train_accuracy, train_f1 = train(train_loader, model, criterion, optimizer, device)
        # Validation step
        val_loss, val_accuracy, val_f1 = validate(val_loader, model, criterion, device)
        
        # Append metrics to history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)
        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)
        
        print(f"Epoch {epoch}/{max_epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        # Check for improvement
        if best_val_loss - val_loss > delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # save the best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'in_channels': int(num_node_features),
                    'out_channels': int(best_params['out_channels']),
                    'num_heads': int(best_params['num_heads']),
                    'batch_size': int(batch_size),
                    'learning_rate': float(best_params['learning_rate']),
                    'weight_decay': float(best_params['weight_decay']),
                }
            },  f'best_model_{dataset_name}.pth')
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
        
    # Plot training and validation curves in one figure with three subplots
    epochs = list(range(1, len(train_loss_history) + 1))
    # Convert metrics to pandas Series for rolling
    train_loss_series = pd.Series(train_loss_history)
    val_loss_series = pd.Series(val_loss_history)
    train_accuracy_series = pd.Series(train_accuracy_history)
    val_accuracy_series = pd.Series(val_accuracy_history)
    train_f1_series = pd.Series(train_f1_history)
    val_f1_series = pd.Series(val_f1_history)
    
    window_size = 3

    # Calculate exponential moving averages
    train_loss_ema = train_loss_series.ewm(span=window_size, adjust=False).mean()
    val_loss_ema = val_loss_series.ewm(span=window_size, adjust=False).mean()
    train_accuracy_ema = train_accuracy_series.ewm(span=window_size, adjust=False).mean()
    val_accuracy_ema = val_accuracy_series.ewm(span=window_size, adjust=False).mean()
    train_f1_ema = train_f1_series.ewm(span=window_size, adjust=False).mean()
    val_f1_ema = val_f1_series.ewm(span=window_size, adjust=False).mean()

    # Create a figure with three subplots in one row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Loss with EMA on the first subplot
    axes[0].plot(epochs, train_loss_history, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_loss_history, label='Validation Loss', marker='o')
    axes[0].plot(epochs, train_loss_ema, label=f'Train Loss EMA (span={window_size})', linestyle='--')
    axes[0].plot(epochs, val_loss_ema, label=f'Validation Loss EMA (span={window_size})', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss with EMA over Epochs')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracy with EMA on the second subplot
    axes[1].plot(epochs, train_accuracy_history, label='Train Accuracy', marker='o')
    axes[1].plot(epochs, val_accuracy_history, label='Validation Accuracy', marker='o')
    axes[1].plot(epochs, train_accuracy_ema, label=f'Train Accuracy EMA (span={window_size})', linestyle='--')
    axes[1].plot(epochs, val_accuracy_ema, label=f'Validation Accuracy EMA (span={window_size})', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy with EMA over Epochs')
    axes[1].legend()
    axes[1].grid(True)

    # Plot F1 Score with EMA on the third subplot
    axes[2].plot(epochs, train_f1_history, label='Train F1 Score', marker='o')
    axes[2].plot(epochs, val_f1_history, label='Validation F1 Score', marker='o')
    axes[2].plot(epochs, train_f1_ema, label=f'Train F1 EMA (span={window_size})', linestyle='--')
    axes[2].plot(epochs, val_f1_ema, label=f'Validation F1 EMA (span={window_size})', linestyle='--')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Training and Validation F1 Score with EMA over Epochs')
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Clean up
    del model
    del optimizer
    del criterion
    torch.cuda.empty_cache()
