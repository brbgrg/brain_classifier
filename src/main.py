
# main.py

import torch
import wandb
from test import test_model
from ingestion import prepare_datasets
from model import GAT
from train_and_test import train, validate
from run_sweep import run_sweep
from sweep_config import sweep_config
import os



# Base directory where your data is stored
base_dir = os.getcwd()

# Prepare datasets
graphs_sc, labels_sc, graphs_sc_combined, labels_sc_combined, feature_names = prepare_datasets(base_dir)

# Run sweeps on both datasets
run_sweep(graphs_sc, labels_sc, sweep_config, dataset_name='graphs_sc')
run_sweep(graphs_sc_combined, labels_sc_combined, sweep_config, dataset_name='graphs_sc_combined')

