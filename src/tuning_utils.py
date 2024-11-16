import torch
import wandb
from refine_sweep_config import build_dataloaders, build_optimizer
from model import GAT


def overfit_small_sample(graphs, labels, sweep_config, dataset_name, num_batches):
    # Modified run_sweep function to limit training batches
    def run_sweep_overfit(graphs, labels, sweep_config, dataset_name, num_batches):
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=f'graph-classification-{dataset_name}')
        
        # Function to execute a single run
        def train_sweep():
            wandb.init()
            config = wandb.config

            # Build data loaders
            train_loader, val_loader, _ = build_dataloaders(graphs, labels, config)
            
            # Limit train_loader to num_batches
            small_train_loader = []
            for i, data in enumerate(train_loader):
                if i >= num_batches:
                    break
                small_train_loader.append(data)
            
            # Model, criterion, and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = GAT(
                in_channels=config.in_channels, 
                out_channels=config.out_channels, 
                num_heads=config.num_heads, 
                num_classes=2
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
            
            # Watch the model with wandb
            wandb.watch(model, log="all", log_freq=10)

            # Train the model on the small dataset
            model.train()
            for epoch in range(config.num_epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                for data in small_train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, data.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += data.y.size(0)
                    correct += (predicted == data.y).sum().item()
                
                train_accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {total_loss/num_batches:.4f}, Accuracy: {train_accuracy:.2f}%")

                # Log metrics
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': total_loss / num_batches,
                    'train_accuracy': train_accuracy,
                })
            
            # Finish the wandb run
            wandb.finish()
        
        # Run the sweep agent
        wandb.agent(sweep_id, function=train_sweep)
        return sweep_id
    
    # Call the modified run_sweep function
    return run_sweep_overfit(graphs, labels, sweep_config, dataset_name, num_batches)