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
