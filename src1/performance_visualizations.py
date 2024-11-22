import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import torch


def plot_hyperparameter_boxplots(results_df, hyperparameters, metric='val_f1', use_hue=True):
    fig, axes = plt.subplots(len(hyperparameters), len(hyperparameters) - 1 if use_hue else 1, figsize=(20, 6 * len(hyperparameters)))
    if len(hyperparameters) == 1:
        axes = np.array([axes])
    for i, param in enumerate(hyperparameters):
        if use_hue:
            for j, hue_param in enumerate([p for p in hyperparameters if p != param]):
                sns.boxplot(x=param, y=metric, data=results_df, hue=hue_param, palette='Set3', ax=axes[i, j])
                axes[i, j].set_title(f'{metric} vs. {param} (Hue: {hue_param})')
                axes[i, j].set_xlabel(param)
                axes[i, j].set_ylabel(metric)
                axes[i, j].legend(title=hue_param)
        else:
            sns.boxplot(x=param, y=metric, data=results_df, ax=axes[i])
            axes[i].set_title(f'{metric} vs. {param}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
    plt.tight_layout()
    plt.show()


def get_top_results(results_df, hyperparameters, metric='val_f1', percentile=10, print_flag=False):
    """
    Fetches the top percentile of runs based on the specified metric.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the grid search.
        hyperparameters (list, optional): List of hyperparameters to include in the output.
        metric (str, optional): The metric to sort by. Defaults to 'val_f1'.
        percentile (float, optional): The top percentile to select (e.g., 10 for top 10%). Defaults to 10.
        print_flag (bool, optional): Whether to print the top runs. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the top percentile of runs.
    """
    selected_columns = ['fold', metric] + hyperparameters
    top_df = results_df[selected_columns].copy()
    
    # Drop NaN in metric
    #top_df.dropna(subset=[metric], inplace=True)
    
    # Sort the dataframe by metric in descending order
    top_df.sort_values(by=metric, ascending=False, inplace=True)
    
    # Filter by percentile if provided
    if percentile is not None:
        threshold = np.percentile(top_df[metric], 100 - percentile)
        top_results_df = top_df[top_df[metric] >= threshold]
    else:
        top_results_df = top_df
    
    if print_flag:
        # Round hyperparameter values to 3 decimal places
        top_results_df[hyperparameters] = top_results_df[hyperparameters].round(3)
        # Display the DataFrame
        display(top_results_df)
    return top_results_df

def plot_hyperparameter_heatmaps(results_df, hyperparameters, metric='val_f1'):
    """
    Plots heatmaps for pairwise combinations of hyperparameters to show the influence on the specified metric.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the grid search.
        hyperparameters (list): List of hyperparameters to include in the plots.
        metric (str, optional): The metric to visualize. Defaults to 'val_f1'.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import combinations

    comb = list(combinations(hyperparameters, 2))
    num_combinations = len(comb)
    cols = 3  # Number of columns for subplots
    rows = (num_combinations + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for ax, (param1, param2) in zip(axes, comb):
        df = results_df.copy()
        df[param1] = df[param1].astype(str)
        df[param2] = df[param2].astype(str)
        pivot_table = df.pivot_table(index=param1, columns=param2, values=metric, aggfunc='mean')
        sns.heatmap(pivot_table, ax=ax, cmap='viridis', annot=True, fmt='.2f')
        ax.set_title(f'{metric} for {param1} vs {param2}')
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)

    # Hide any unused subplots
    for i in range(len(comb), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()




def plot_uniform_hyperparameter_heatmaps(results_df, hyperparameters, metric='val_f1'):
    """
    Plots heatmaps for pairwise combinations of hyperparameters to show the influence on the specified metric,
    using a uniform color scale across all subplots.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the grid search.
        hyperparameters (list): List of hyperparameters to include in the plots.
        metric (str, optional): The metric to visualize. Defaults to 'val_f1'.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import combinations

    comb = list(combinations(hyperparameters, 2))
    num_combinations = len(comb)
    cols = 3  # Number of columns for subplots
    rows = (num_combinations + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    # Compute global min and max for color scaling
    min_val = results_df[metric].min()
    max_val = results_df[metric].max()

    for ax, (param1, param2) in zip(axes, comb):
        df = results_df.copy()
        df[param1] = df[param1].astype(str)
        df[param2] = df[param2].astype(str)
        pivot_table = df.pivot_table(index=param1, columns=param2, values=metric, aggfunc='mean')
        sns.heatmap(
            pivot_table, ax=ax, cmap='viridis', annot=True, fmt='.2f',
            vmin=min_val, vmax=max_val
        )
        ax.set_title(f'{metric} for {param1} vs {param2}')
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)

    # Hide any unused subplots
    for i in range(len(comb), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()




def plot_hyperparameter_scatterplots_with_regression(results_df, hyperparameters, metric='val_f1'):
    """
    Plots scatterplots with regression lines for specified hyperparameters against the metric.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the grid search.
        hyperparameters (list): List of hyperparameters to plot.
        metric (str, optional): The metric to plot against. Defaults to 'val_f1'.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    num_hyperparams = len(hyperparameters)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_hyperparams + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for ax, param in zip(axes, hyperparameters):
        sns.regplot(x=param, y=metric, data=results_df, ax=ax, scatter_kws={'alpha': 0.5})
        ax.set_title(f'{metric} vs {param}')
        ax.set_xlabel(param)
        ax.set_ylabel(metric)

    # Hide any unused subplots
    for i in range(num_hyperparams, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_parallel_coordinates(results_df, hyperparameters, metric='val_f1', top_percentile=None):
    """
    Plots a parallel coordinates plot for the visualization of the whole grid.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the grid search.
        hyperparameters (list): List of hyperparameters to include in the plot.
        metric (str, optional): The metric to visualize. Defaults to 'val_f1'.
        top_percentile (float, optional): If specified, only include the top percentile of runs based on the metric.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas.plotting import parallel_coordinates
    from sklearn.preprocessing import MinMaxScaler

    # Select the columns to include
    cols = hyperparameters + [metric]
    df = results_df[cols].copy()

    # If top_percentile is specified, filter the DataFrame
    if top_percentile is not None:
        threshold = np.percentile(df[metric], 100 - top_percentile)
        df = df[df[metric] >= threshold]

    # Normalize hyperparameters and metric for plotting
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols)

    # Add a class label based on the metric for coloring
    num_classes = 4  # Adjust as needed
    df_scaled['metric_class'] = pd.qcut(df[metric], num_classes, labels=False)
    df_scaled.dropna(subset=['metric_class'], inplace=True)

    plt.figure(figsize=(12, 6))
    parallel_coordinates(df_scaled, 'metric_class', cols=cols, color=plt.cm.Set1.colors)
    plt.title('Parallel Coordinates Plot')
    plt.xlabel('Hyperparameters and Metric')
    plt.ylabel('Scaled Value')
    plt.legend(title='Metric Class', loc='upper right')
    plt.show()


