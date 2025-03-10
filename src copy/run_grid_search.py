# run_grid_search.py

from fetch_and_visualize import fetch_runs_dataframe, get_top_runs, tag_top_runs
from run_sweep import run_sweep
from IPython.display import display

def run_grid_search(graphs, labels, sweep_config, dataset_name, top_x=10, sweep_count=None):
    # Run the sweep
    sweep_id = run_sweep(graphs, labels, sweep_config, dataset_name, sweep_count=sweep_count)
    
    # Fetch and analyze results
    project_name = f'graph-classification-{dataset_name}'
    all_runs_df = fetch_runs_dataframe(project_name, sweep_id=sweep_id)
    
    # Display top runs
    top_runs_df = get_top_runs(all_runs_df, metric='val_f1', top_x=top_x)
    
    print(f"\nTop {top_x} runs for {dataset_name}:")
    display(top_runs_df[['run_id', 'run_name', 'val_f1', 'learning_rate', 'out_channels', 'num_heads', 'weight_decay']])
    
    # Tag the top runs in wandb
    tag_top_runs(top_runs_df, project_name)
    
    # Return the dataframes for further analysis
    return all_runs_df, top_runs_df