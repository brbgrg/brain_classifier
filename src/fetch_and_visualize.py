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
    plt.xscale('log' if 'learning_rate' in hyperparameter else 'linear')
    plt.title(f'{hyperparameter} vs. {metric}')
    plt.show()

def compute_correlation(df, hyperparameters, metric):
    corr = df[hyperparameters + [metric]].corr()
    return corr[metric].drop(metric)

def tag_top_runs(top_runs_df, project_name, entity_name=None):
    api = wandb.Api()
    for _, row in top_runs_df.iterrows():
        run_id = row['run_id']
        if entity_name:
            run_path = f"{entity_name}/{project_name}/{run_id}"
        else:
            run_path = f"{project_name}/{run_id}"
        run = api.run(run_path)
        tags = run.tags
        if 'top_performing' not in tags:
            tags.append('top_performing')
            run.tags = tags
            run.update()