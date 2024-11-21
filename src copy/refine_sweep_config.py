import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fetch_runs_dataframe(project_name, entity_name=None, sweep_id=None):
    api = wandb.Api()
    if entity_name:
        runs = api.runs(f'{entity_name}/{project_name}')
    else:
        runs = api.runs(project_name)
    
    summary_list = [] 
    config_list = [] 
    name_list = [] 

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

    summary_df = pd.DataFrame(summary_list) 
    config_df = pd.DataFrame(config_list) 
    name_df = pd.DataFrame({'run_name': name_list}) 
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)
    return all_df

def plot_hyperparameter_performance(df, hyperparameter, metric):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=hyperparameter, y=metric)
    plt.xscale('log' if 'learning_rate' in hyperparameter else 'linear')
    plt.title(f'{hyperparameter} vs. {metric}')
    plt.show()

def get_top_runs(df, metric, top_percent=10):
    threshold = df[metric].quantile(1 - top_percent / 100)
    top_runs = df[df[metric] >= threshold]
    return top_runs

def compute_correlation(df, hyperparameters, metric):
    corr = df[hyperparameters + [metric]].corr()
    return corr[metric].drop(metric)

def update_numeric_hyperparameter(sweep_config, df, hyperparameter, metric, top_percent=10):
    top_runs = get_top_runs(df, metric, top_percent)
    min_value = top_runs[hyperparameter].min()
    max_value = top_runs[hyperparameter].max()
    
    # Update the sweep configuration
    param_config = sweep_config['parameters'][hyperparameter]
    if 'distribution' in param_config:
        distribution = param_config['distribution']
        sweep_config['parameters'][hyperparameter] = {
            'distribution': distribution,
            'min': float(min_value),
            'max': float(max_value)
        }
    else:
        sweep_config['parameters'][hyperparameter]['min'] = float(min_value)
        sweep_config['parameters'][hyperparameter]['max'] = float(max_value)
    return sweep_config


def update_categorical_hyperparameter(sweep_config, df, hyperparameter, metric, top_percent=10):
    top_runs = get_top_runs(df, metric, top_percent)
    top_values = top_runs[hyperparameter].value_counts().index.tolist()
    
    # Update the sweep configuration
    if len(top_values) == 1:
        sweep_config['parameters'][hyperparameter] = {'value': top_values[0]}
    else:
        sweep_config['parameters'][hyperparameter]['values'] = top_values
    return sweep_config


"""
def exclude_poor_hyperparameter_values(sweep_config, df, hyperparameter, metric, bottom_percent=10):
    bottom_runs = df.nsmallest(int(len(df) * (bottom_percent / 100)), metric)
    poor_values = bottom_runs[hyperparameter].unique()
    
    # Exclude poor values from the sweep configuration
    current_values = sweep_config['parameters'][hyperparameter]['values']
    updated_values = [val for val in current_values if val not in poor_values]
    sweep_config['parameters'][hyperparameter]['values'] = updated_values
    return sweep_config
"""

def refine_sweep_config(sweep_config, all_runs_df, metric='val_f1', top_percent=10):
    # List of hyperparameters to update
    numeric_hyperparameters = ['learning_rate', 'weight_decay']
    categorical_hyperparameters = ['optimizer', 'batch_size', 'out_channels', 'num_heads']
    
    # Update numeric hyperparameters
    for param in numeric_hyperparameters:
        sweep_config = update_numeric_hyperparameter(sweep_config, all_runs_df, param, metric, top_percent)
    
    # Update categorical hyperparameters
    for param in categorical_hyperparameters:
        sweep_config = update_categorical_hyperparameter(sweep_config, all_runs_df, param, metric, top_percent)
    
    return sweep_config
