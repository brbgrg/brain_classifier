# sweep.py

import torch
import torch.nn as nn
import wandb
from model import GAT  # Import the model from model.py
from model import GAT
from train_and_test import train, validate
from data_utils import build_optimizer, build_dataloaders

def run_sweep(graphs, labels, sweep_config, dataset_name):
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='graph-classification-' + dataset_name)
    # Function to execute a single run
    def train_sweep():
        wandb.init()
        config = wandb.config

        # Build data loaders
        train_loader, val_loader, _ = build_dataloaders(graphs, labels, config)

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
                    'config': config
                }, f'best_model_{dataset_name}.pth')
                wandb.save(f'best_model_{dataset_name}.pth')

        # Finish the wandb run
        wandb.finish()

    wandb.agent(sweep_id, function=train_sweep)
