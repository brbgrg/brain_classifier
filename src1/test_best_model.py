# test.py
import torch
from model import GAT  # Import the model here
from train_test import test
from build_dataloader import build_dataloader, set_seed, GraphDataset
from types import SimpleNamespace

def test_model(model_path, graphs, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = SimpleNamespace(**checkpoint['config']) 

    # Initialize the model
    model = GAT(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_heads=config.num_heads,
        num_classes=2
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = GraphDataset(graphs, labels) 
    test_loader = build_dataloader(test_dataset, config.batch_size, shuffle=False)

    # Test the model
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_f1 = test(test_loader, model, criterion, device)

    # Print the test metrics
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")