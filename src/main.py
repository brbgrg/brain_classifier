# main.py

from test_model import test_model
from ingestion import prepare_datasets
from incremental_tuning import incremental_tuning
from sweep_config import sweep_config
import os
import copy

# Base directory where your data is stored
base_dir = os.getcwd()

# Prepare datasets
graphs_sc, labels_sc, graphs_sc_combined, labels_sc_combined, feature_names = prepare_datasets(base_dir)

# Load best models ...
# Model paths
model_path_sc = r'C:\Users\barbo\brain classifier repo\brain_classifier\src\best_model_graphs_sc.pth'

# Test the model on the test set
test_model(model_path_sc, graphs_sc, labels_sc)

# Test the model on the test set
#test_model(model_path_sc_combined, graphs_sc_combined, labels_sc_combined)
