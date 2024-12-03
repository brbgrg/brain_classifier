# test.py
import torch
from model import GAT  # Import the model here
from train_test import test
from build_dataloader import build_dataloader, set_seed, GraphDataset
from prepare_datasets import normalize_graph_features, normalize_graph_edge_weights
from types import SimpleNamespace

def test_model(model_path, graphs, labels, model_class, train_mean_x, train_std_x, train_mean_edge_attr, train_std_edge_attr, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = SimpleNamespace(**checkpoint['config']) 

    # Initialize the model
    model = model_class(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_heads = getattr(config, 'num_heads', 1),
        num_classes=2
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    test_graphs = normalize_graph_features(graphs, train_mean_x, train_std_x)
    test_graphs = normalize_graph_edge_weights(test_graphs, train_mean_edge_attr, train_std_edge_attr)
    test_dataset = GraphDataset(test_graphs, labels)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test the model
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_f1 = test(test_loader, model, criterion, device)

    # Print the test metrics
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
