# sweep_config.py
sweep_config = {
    'method': 'random',  # Choose 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd'] #usa solo adam
        },
        'test_size': {
            'value': 0.3
        },
        'learning_rate': {
            #'values': [1e-3, 1e-4, 1e-5]
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'out_channels': {
            'values': [4, 8, 12, 16]
        },
        'num_heads': {
            'values': [1, 2, 3, 4, 5]
        },
        'num_epochs': {
            'value': 1
        },
        'weight_decay': {
            'values': [0, 1e-5, 1e-4, 1e-3, 1e-2]
        },
        'batch_size': {
            'values': [8, 16, 32] #usa solo 32
        },
        'random_state': {
            'value': 42
        }
    }
}

# Calculate the size of the search space
"""
search_space_size = (
    len(sweep_config['parameters']['optimizer']['values']) *
    len(sweep_config['parameters']['learning_rate']['values']) *
    len(sweep_config['parameters']['out_channels']['values']) *
    len(sweep_config['parameters']['num_heads']['values']) *
    len(sweep_config['parameters']['weight_decay']['values']) *
    len(sweep_config['parameters']['batch_size']['values'])
)

print(f"Search space size: {search_space_size}")
"""
