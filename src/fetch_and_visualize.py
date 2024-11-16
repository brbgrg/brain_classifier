import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


def fetch_runs_dataframe(project_name, entity_name=None, sweep_id=None):
    api = wandb.Api()
    if entity_name:
        runs = api.runs(f'{entity_name}/{project_name}')
    else:
        runs = api.runs(project_name)
    
    summary_list = [] 
    config_list = [] 
    name_list = [] 
    run_id_list = []

    for run in runs: 
        # Exclude any runs that are not finished
        if run.state != 'finished':
            continue
        # Filter by sweep_id if provided
        if sweep_id and run.sweep.id != sweep_id:
            continue
        summary_list.append(run.summary._json_dict) 
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 
        name_list.append(run.name)
        run_id_list.append(run.id)

    summary_df = pd.DataFrame(summary_list) 
    config_df = pd.DataFrame(config_list) 
    name_df = pd.DataFrame({'run_name': name_list}) 
    run_id_df = pd.DataFrame({'run_id': run_id_list})
    all_df = pd.concat([name_df, run_id_df, config_df, summary_df], axis=1)
    return all_df

def get_top_runs(df, metric, top_x=10):
    top_runs = df.nlargest(top_x, metric)
    return top_runs

def plot_hyperparameter_performance(df, hyperparameter, metric):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=hyperparameter, y=metric)
    plt.xscale('linear')
    plt.title(f'{hyperparameter} vs. {metric}')
    plt.show()

def compute_correlation(df, hyperparameters, metric):
    corr = df[hyperparameters + [metric]].corr()
    return corr[metric].drop(metric)

def tag_top_runs(top_runs_df, project_name, entity_name=None):
    api = wandb.Api()
    top_run_ids = set(top_runs_df['run_id'])
    runs = api.runs(f"{entity_name}/{project_name}" if entity_name else project_name)
    
    for run in runs:
        tags = set(run.tags)
        if run.id in top_run_ids:
            tags.add('top_performing')
        else:
            tags.discard('top_performing')
        run.tags = list(tags)
        run.save()


def plot_train_val_curves(run_id, metrics, entity_name=None, project_name=None):
    """
    Plots training and validation curves for specified metrics over all epochs.

    Args:
        run_id (str): The ID of the wandb run.
        metrics (list): The list of metrics to plot (e.g., ['accuracy', 'loss']).
        entity_name (str, optional): The wandb entity/team name. Defaults to None.
        project_name (str, optional): The wandb project name. Defaults to None.

    Returns:
        None: Displays the plot.
    """
    api = wandb.Api()

    if entity_name and project_name:
        run_path = f"{entity_name}/{project_name}/{run_id}"
    else:
        run_path = f"{project_name}/{run_id}"

    try:
        run = api.run(run_path)
    except wandb.errors.CommError:
        print(f"Run {run_path} not found.")
        return

    history = run.history(keys=[f'train_{metric}' for metric in metrics] + [f'val_{metric}' for metric in metrics] + ['epoch'])

    if history.empty:
        print(f"No data found for metrics '{metrics}' in run {run_path}.")
        return

    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 4 * num_metrics), sharey=True)
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.lineplot(data=history, x='epoch', y=f'train_{metric}', label='Train', marker='o', ax=ax)
        sns.lineplot(data=history, x='epoch', y=f'val_{metric}', label='Validation', marker='o', ax=ax)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Train vs Validation {metric.capitalize()}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

