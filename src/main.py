
# main.py

from test import test_model
from ingestion import prepare_datasets
from run_sweep import run_sweep
from sweep_config import sweep_config
import os
import copy

# Base directory where your data is stored
base_dir = os.getcwd()

# Prepare datasets
graphs_sc, labels_sc, graphs_sc_combined, labels_sc_combined, feature_names = prepare_datasets(base_dir)

# Create a copy of the sweep config
sweep_config_sc = copy.deepcopy(sweep_config)
sweep_config_sc_combined = copy.deepcopy(sweep_config)

# Run sweeps on both datasets
run_sweep(graphs_sc, labels_sc, sweep_config, dataset_name='graphs_sc')
run_sweep(graphs_sc_combined, labels_sc_combined, sweep_config, dataset_name='graphs_sc_combined')

# Run the model on the test set
model_path_sc = 'best_model_sc.pth'
model_path_sc_combined = 'best_model_sc_combined.pth'

# Refine sweep config ...

# Test the model on the test set
test_model(model_path_sc, graphs_sc, labels_sc)

# Test the model on the test set
test_model(model_path_sc_combined, graphs_sc_combined, labels_sc_combined)