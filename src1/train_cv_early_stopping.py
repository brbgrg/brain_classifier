import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_test import train, validate
from build_dataloader import build_dataloader, set_seed, GraphDataset
from model import GAT
import pandas as pd
import numpy as np  # Make sure to import numpy
from sklearn.model_selection import StratifiedKFold

def train_model_with_early_stopping_cv(graphs, labels, dataset_name, best_params, patience, max_epochs, batch_size, device, num_splits, model_save_path):
    set_seed(42)
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    num_node_features = graphs[0].x.size(1)

    all_train_loss_history = []
    all_val_loss_history = []
    all_train_accuracy_history = []
    all_val_accuracy_history = []
    all_train_f1_history = []
    all_val_f1_history = []

    overall_best_val_loss = float('inf')
    overall_best_model_state = None

    for fold, (train_index, val_index) in enumerate(kf.split(graphs, labels)):
        print(f"Fold {fold + 1}/{num_splits}")
        train_graphs_fold = [graphs[i] for i in train_index]
        val_graphs_fold = [graphs[i] for i in val_index]
        train_labels_fold = [labels[i] for i in train_index]
        val_labels_fold = [labels[i] for i in val_index]

        train_dataset = GraphDataset(train_graphs_fold, train_labels_fold)
        val_dataset = GraphDataset(val_graphs_fold, val_labels_fold)

        train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

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

        best_val_loss_fold = float('inf')
        epochs_without_improvement = 0
        delta = 1e-4

        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []
        train_f1_history = []
        val_f1_history = []

        for epoch in range(1, max_epochs + 1):
            train_loss, train_accuracy, train_f1 = train(train_loader, model, criterion, optimizer, device)
            val_loss, val_accuracy, val_f1 = validate(val_loader, model, criterion, device)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(val_accuracy)
            train_f1_history.append(train_f1)
            val_f1_history.append(val_f1)

            print(f"Epoch {epoch}/{max_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

            if best_val_loss_fold - val_loss > delta:
                best_val_loss_fold = val_loss
                epochs_without_improvement = 0
                # Save the best model for this fold
                best_model_state_fold = model.state_dict()
                best_epoch_fold = epoch
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        # Check if this fold has the overall best model
        if best_val_loss_fold < overall_best_val_loss:
            overall_best_val_loss = best_val_loss_fold
            overall_best_model_state = best_model_state_fold
            overall_best_config = {
                'in_channels': int(num_node_features),
                'out_channels': int(best_params['out_channels']),
                'num_heads': int(best_params['num_heads']),
                'batch_size': int(batch_size),
                'learning_rate': float(best_params['learning_rate']),
                'weight_decay': float(best_params['weight_decay']),
            }
            overall_best_epoch = best_epoch_fold

        num_epochs_this_fold = len(train_loss_history)
        if num_epochs_this_fold < max_epochs:
            pad_length = max_epochs - num_epochs_this_fold
            train_loss_history.extend([np.nan] * pad_length)
            val_loss_history.extend([np.nan] * pad_length)
            train_accuracy_history.extend([np.nan] * pad_length)
            val_accuracy_history.extend([np.nan] * pad_length)
            train_f1_history.extend([np.nan] * pad_length)
            val_f1_history.extend([np.nan] * pad_length)

        all_train_loss_history.append(train_loss_history)
        all_val_loss_history.append(val_loss_history)
        all_train_accuracy_history.append(train_accuracy_history)
        all_val_accuracy_history.append(val_accuracy_history)
        all_train_f1_history.append(train_f1_history)
        all_val_f1_history.append(val_f1_history)

        del model
        del optimizer
        del criterion
        torch.cuda.empty_cache()

    # Save the overall best model
    torch.save({
        'model_state_dict': overall_best_model_state,
        'config': overall_best_config,
        'best_epoch': overall_best_epoch
    }, model_save_path)
    print(f"Best model saved to {model_save_path}")

    all_train_loss_history = np.array(all_train_loss_history, dtype=np.float32)
    all_val_loss_history = np.array(all_val_loss_history, dtype=np.float32)
    all_train_accuracy_history = np.array(all_train_accuracy_history, dtype=np.float32)
    all_val_accuracy_history = np.array(all_val_accuracy_history, dtype=np.float32)
    all_train_f1_history = np.array(all_train_f1_history, dtype=np.float32)
    all_val_f1_history = np.array(all_val_f1_history, dtype=np.float32)

    avg_train_loss = np.nanmean(all_train_loss_history, axis=0)
    avg_val_loss = np.nanmean(all_val_loss_history, axis=0)
    avg_train_accuracy = np.nanmean(all_train_accuracy_history, axis=0)
    avg_val_accuracy = np.nanmean(all_val_accuracy_history, axis=0)
    avg_train_f1 = np.nanmean(all_train_f1_history, axis=0)
    avg_val_f1 = np.nanmean(all_val_f1_history, axis=0)

    epochs = list(range(1, max_epochs + 1))
    train_loss_series = pd.Series(avg_train_loss)
    val_loss_series = pd.Series(avg_val_loss)
    train_accuracy_series = pd.Series(avg_train_accuracy)
    val_accuracy_series = pd.Series(avg_val_accuracy)
    train_f1_series = pd.Series(avg_train_f1)
    val_f1_series = pd.Series(avg_val_f1)

    window_size = 3

    train_loss_ema = train_loss_series.ewm(span=window_size, adjust=False).mean()
    val_loss_ema = val_loss_series.ewm(span=window_size, adjust=False).mean()
    train_accuracy_ema = train_accuracy_series.ewm(span=window_size, adjust=False).mean()
    val_accuracy_ema = val_accuracy_series.ewm(span=window_size, adjust=False).mean()
    train_f1_ema = train_f1_series.ewm(span=window_size, adjust=False).mean()
    val_f1_ema = val_f1_series.ewm(span=window_size, adjust=False).mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(epochs, avg_train_loss, label='Average Train Loss', marker='o')
    axes[0].plot(epochs, avg_val_loss, label='Average Validation Loss', marker='o')
    axes[0].plot(epochs, train_loss_ema, label=f'Train Loss EMA (span={window_size})', linestyle='--')
    axes[0].plot(epochs, val_loss_ema, label=f'Validation Loss EMA (span={window_size})', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Average Training and Validation Loss with EMA over Epochs')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, avg_train_accuracy, label='Average Train Accuracy', marker='o')
    axes[1].plot(epochs, avg_val_accuracy, label='Average Validation Accuracy', marker='o')
    axes[1].plot(epochs, train_accuracy_ema, label=f'Train Accuracy EMA (span={window_size})', linestyle='--')
    axes[1].plot(epochs, val_accuracy_ema, label=f'Validation Accuracy EMA (span={window_size})', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Average Training and Validation Accuracy with EMA over Epochs')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, avg_train_f1, label='Average Train F1 Score', marker='o')
    axes[2].plot(epochs, avg_val_f1, label='Average Validation F1 Score', marker='o')
    axes[2].plot(epochs, train_f1_ema, label=f'Train F1 EMA (span={window_size})', linestyle='--')
    axes[2].plot(epochs, val_f1_ema, label=f'Validation F1 EMA (span={window_size})', linestyle='--')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Average Training and Validation F1 Score with EMA over Epochs')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
