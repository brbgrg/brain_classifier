# main.py

from test import test_model
from ingestion import prepare_datasets
from incremental_tuning import incremental_tuning
from sweep_config import sweep_config
import os
import copy

# Base directory where your data is stored
base_dir = os.getcwd()

# Prepare datasets
graphs_sc, labels_sc, graphs_sc_combined, labels_sc_combined, feature_names = prepare_datasets(base_dir)

# Create copies of the sweep config for each dataset
sweep_config_sc = copy.deepcopy(sweep_config)
sweep_config_sc_combined = copy.deepcopy(sweep_config)

# Update the sweep config with the in_channels
sweep_config_sc['parameters']['in_channels'] = {
    'value': 6
}

sweep_config_sc_combined['parameters']['in_channels'] = {
    'value': 8
}

"""
# Perform incremental tuning on both datasets
# For graphs_sc
incremental_tuning(
    graphs=graphs_sc,
    labels=labels_sc,
    initial_sweep_config=sweep_config_sc,
    dataset_name='graphs_sc',
    num_iterations=3,
    sweep_count=50  # Adjust 
)
"""

# For graphs_sc_combined
incremental_tuning(
    graphs=graphs_sc_combined,
    labels=labels_sc_combined,
    initial_sweep_config=sweep_config_sc_combined,
    dataset_name='graphs_sc_combined',
    num_iterations=3, #num_iterations of the incremental tuning
    sweep_count=50  # Adjust (tot number of combinations is around 300)
)

# After tuning, run the model on the test set
#model_path_sc = 'best_model_graphs_sc.pth'
model_path_sc_combined = 'best_model_graphs_sc_combined.pth'

# Test the model on the test set
#test_model(model_path_sc, graphs_sc, labels_sc)

# Test the model on the test set
#test_model(model_path_sc_combined, graphs_sc_combined, labels_sc_combined)
