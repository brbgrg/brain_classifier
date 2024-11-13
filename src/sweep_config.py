# sweep_config.py
sweep_config = {
    'method': 'random',  # Choose 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'test_size': {
            'value': 0.3
        },
        'learning_rate': {
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
            'value': 5
        },
        'weight_decay': {
            'values': [0, 1e-6, 1e-5, 1e-4, 1e-3]
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        'random_state': {
            'value': 42
        }
    }
}
