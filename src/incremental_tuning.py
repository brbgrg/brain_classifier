# incremental_tuning.py

from refine_sweep_config import refine_sweep_config, fetch_runs_dataframe
from run_sweep import run_sweep
import copy

def incremental_tuning(graphs, labels, initial_sweep_config, dataset_name, num_iterations=3, sweep_count=50, num_epochs_list=None):
    if num_epochs_list is None:
        num_epochs_list = [5, 10, 20]  # Default values
    sweep_config = copy.deepcopy(initial_sweep_config)
    for i in range(num_iterations):
        # Update num_epochs
        sweep_config['parameters']['num_epochs']['value'] = num_epochs_list[i]
        
        # Run the sweep
        sweep_id = run_sweep(graphs, labels, sweep_config, dataset_name, sweep_count=sweep_count)
        
        # Fetch and analyze results
        project_name = f'graph-classification-{dataset_name}'
        all_runs_df = fetch_runs_dataframe(project_name, sweep_id= sweep_id)
        
        # Refine sweep configuration
        #sweep_config = refine_sweep_config(sweep_config, all_runs_df)

        print(f"Refined hyperparameters after iteration {i+1}: {sweep_config['parameters']}")
        print(f"Completed iteration {i+1}/{num_iterations}")

