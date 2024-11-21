import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import torch


import wandb
import pandas as pd

def fetch_runs_dataframe(project_name, entity_name=None, sweep_id=None):
    api = wandb.Api()
    project_path = f"{entity_name}/{project_name}" if entity_name else project_name
    runs = api.runs(project_path)
    
    runs_data = []
    hyperparameter_columns = set()
    
    for run in runs:
        # Exclude any runs that are not finished
        if run.state != 'finished':
            continue
        # Filter by sweep_id if provided
        if sweep_id and (run.sweep is None or run.sweep.id != sweep_id):
            continue
        # Collect data
        run_data = {
            'run_name': run.name,
            'run_id': run.id
        }
        # Filter out internal config keys starting with '_'
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        run_data.update(config)
        run_data.update(run.summary._json_dict)
        runs_data.append(run_data)
        hyperparameter_columns.update(config.keys())
    
    all_df = pd.DataFrame(runs_data)
    # Drop rows where any hyperparameter is NaN
    all_df.dropna(subset=hyperparameter_columns, inplace=True)
    return all_df

def fetch_top_runs(project_name, sweep_id=None, hyperparameters=None, metric='val_f1', percentile=None, print_flag=False):
    all_df = fetch_runs_dataframe(project_name=project_name, entity_name=None, sweep_id=sweep_id)
    # Ensure hyperparameters is a list
    hyperparameters = hyperparameters or []
    selected_columns = ['run_id', 'run_name', metric] + hyperparameters
    all_df = all_df[selected_columns]
    
    # Drop NaN in metric
    all_df.dropna(subset=[metric], inplace=True)
    
    # Sort the dataframe by metric in descending order
    all_df.sort_values(by=metric, ascending=False, inplace=True)
    
    # Filter by percentile if provided
    if percentile is not None:
        threshold = all_df[metric].quantile(percentile / 100)
        top_runs_df = all_df[all_df[metric] >= threshold]
    else:
        top_runs_df = all_df

    if print_flag:
        # Round hyperparameter values to 3 decimal places
        if hyperparameters:
            top_runs_df[hyperparameters] = top_runs_df[hyperparameters].round(3)
        # Display the DataFrame
        display(top_runs_df.style.background_gradient(cmap='YlGn', subset=[metric]))
    return top_runs_df


def plot_hyperparameter_boxplots(sweep_id, hyperparameters, project_name, entity_name=None, metric='val_f1', use_hue=True):
    all_df = fetch_runs_dataframe(project_name=project_name, entity_name=entity_name, sweep_id=sweep_id)
    fig, axes = plt.subplots(len(hyperparameters), len(hyperparameters) - 1 if use_hue else 1, figsize=(20, 6 * len(hyperparameters)))
    for i, param in enumerate(hyperparameters):
        if use_hue:
            for j, hue_param in enumerate([p for p in hyperparameters if p != param]):
                sns.boxplot(x=param, y=metric, data=all_df, hue=hue_param, palette='Set3', ax=axes[i, j])
                axes[i, j].set_title(f'{metric} vs. {param} (Hue: {hue_param})')
                axes[i, j].set_xlabel(param)
                axes[i, j].set_ylabel(metric)
                axes[i, j].legend(title=hue_param)
        else:
            sns.boxplot(x=param, y=metric, data=all_df, ax=axes[i])
            axes[i].set_title(f'{metric} vs. {param}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
    plt.tight_layout()
    plt.show()


from itertools import combinations

def plot_hyperparameter_heatmaps(sweep_id, hyperparameters, project_name, entity_name=None, metric='val_f1'):
    all_df = fetch_runs_dataframe(project_name=project_name, entity_name=entity_name, sweep_id=sweep_id)
    comb = list(combinations(hyperparameters, 2))
    num_combinations = len(comb)
    cols = 3  # Define number of columns for subplots
    rows = (num_combinations + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    
    for ax, (param1, param2) in zip(axes, comb):
        all_df[param1] = all_df[param1].astype(str)
        all_df[param2] = all_df[param2].astype(str)
        pivot_table = all_df.pivot_table(index=param1, columns=param2, values=metric, aggfunc='mean')
        sns.heatmap(pivot_table, ax=ax, cmap='viridis', annot=True, fmt='.2f')
        ax.set_title(f'{metric} for {param1} vs {param2}')
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)
    
    # Remove any unused subplots
    for ax in axes[len(comb):]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()


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


def plot_train_val_curves(run_id, metrics, entity_name=None, project_name=None, display_config=False):
    """
    Plots training and validation curves for specified metrics over all epochs.

    Args:
        run_id (str): The ID of the wandb run.
        metrics (list): The list of metrics to plot (e.g., ['accuracy', 'loss']).
        entity_name (str, optional): The wandb entity/team name. Defaults to None.
        project_name (str, optional): The wandb project name. Defaults to None.
        display_config (bool, optional): If True, display the hyperparameters configuration. Defaults to False.

    Returns:
        None: Displays the plot.
    """
    api = wandb.Api()

    if entity_name is not None and project_name is not None:
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
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4), sharex=True)
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
        window_size = 10
        train_ma = pd.Series(history[f'train_{metric}']).rolling(window=window_size).mean()
        val_ma = pd.Series(history[f'val_{metric}']).rolling(window=window_size).mean()
        sns.lineplot(x=history['epoch'], y=train_ma, label=f'Train MA (window={window_size})', ax=ax)
        sns.lineplot(x=history['epoch'], y=val_ma, label=f'Validation MA (window={window_size})', ax=ax)

    plt.tight_layout()
    plt.show()
    
    if display_config:
        config = run.config
        print("Hyperparameters configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")


def plot_train_val_curves_path(file_path, metrics):
    """
    Plots training and validation curves for specified metrics over all epochs from a saved model file.

    Args:
        file_path (str): The path to the saved model file.
        metrics (list): The list of metrics to plot (e.g., ['accuracy', 'loss']).

    Returns:
        None: Displays the plot.
    """
    # Load the checkpoint
    checkpoint = torch.load(file_path)
    
    # Extract the config and history
    config = checkpoint['config']
    history = checkpoint.get('history', None)
    
    if history is None:
        print(f"No history data found in file {file_path}.")
        return

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4), sharex=True)
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_metric = [epoch_data[f'train_{metric}'] for epoch_data in history]
        val_metric = [epoch_data[f'val_{metric}'] for epoch_data in history]
        epochs = list(range(1, len(train_metric) + 1))

        sns.lineplot(x=epochs, y=train_metric, label='Train', marker='o', ax=ax)
        sns.lineplot(x=epochs, y=val_metric, label='Validation', marker='o', ax=ax)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Train vs Validation {metric.capitalize()}')
        ax.legend()
        ax.grid(True)
        window_size = 3
        train_ma = pd.Series(train_metric).rolling(window=window_size).mean()
        val_ma = pd.Series(val_metric).rolling(window=window_size).mean()
        sns.lineplot(x=epochs, y=train_ma, label=f'Train MA (window={window_size})', ax=ax)
        sns.lineplot(x=epochs, y=val_ma, label=f'Validation MA (window={window_size})', ax=ax)

    plt.tight_layout()
    plt.show()


def get_config(model_path=None, run_id=None, entity_name=None, project_name=None):
    import torch
    api = wandb.Api()

    if model_path:
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
    elif run_id:
        if entity_name is not None and project_name is not None:
            run_path = f"{entity_name}/{project_name}/{run_id}"
        else:
            run_path = f"{project_name}/{run_id}"

        try:
            run = api.run(run_path)
            config = run.config
        except wandb.errors.CommError:
            print(f"Run {run_path} not found.")
            return None
    else:
        print("Either model_path or run_id must be provided.")
        return None

    # Print config in a column
    df = pd.DataFrame(list(config.items()), columns=['Parameter', 'Value'])
    print(df.to_string(index=False))

    return config


def plot_confusion_matrix(run_id, entity_name=None, project_name=None):
    api = wandb.Api()

    if entity_name is not None and project_name is not None:
        run_path = f"{entity_name}/{project_name}/{run_id}"
    else:
        run_path = f"{project_name}/{run_id}"

    try:
        run = api.run(run_path)
    except wandb.errors.CommError:
        print(f"Run {run_path} not found.")
        return

    confusion_matrix = run.summary.get('confusion_matrix')
    if confusion_matrix is None:
        print(f"No confusion matrix found for run {run_path}.")
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    display(run.summary.get('classification_report'))


def plot_roc_curve(run_id, entity_name=None, project_name=None):
    api = wandb.Api()

    if entity_name is not None and project_name is not None:
        run_path = f"{entity_name}/{project_name}/{run_id}"
    else:
        run_path = f"{project_name}/{run_id}"

    try:
        run = api.run(run_path)
    except wandb.errors.CommError:
        print(f"Run {run_path} not found.")
        return

    roc_auc = run.summary.get('roc_auc')
    if roc_auc is None:
        print(f"No ROC AUC found for run {run_path}.")
        return

    fpr = run.summary.get('fpr')
    tpr = run.summary.get('tpr')

    plt.figure(figsize=(8, 6))
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pr_curve(run_id, entity_name=None, project_name=None):
    
    api = wandb.Api()

    if entity_name is not None and project_name is not None:
        run_path = f"{entity_name}/{project_name}/{run_id}"
    else:
        run_path = f"{project_name}/{run_id}"

    try:
        run = api.run(run_path)
    except wandb.errors.CommError:
        print(f"Run {run_path} not found.")
        return

    pr_auc = run.summary.get('pr_auc')
    if pr_auc is None:
        print(f"No PR AUC found for run {run_path}.")
        return

    precision = run.summary.get('precision')
    recall = run.summary.get('recall')

    plt.figure(figsize=(8, 6))
    for i in range(len(precision)):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AUC = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


