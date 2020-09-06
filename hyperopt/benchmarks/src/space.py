from hyperopt import hp


space = {
    'adadelta': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'rho': 1 - hp.loguniform('rho', -4, 0),
        'epsilon': hp.loguniform('epsilon', -8, -4)
    },
    'adagrad': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'initial_accumulator_value': 1 - hp.loguniform('initial_accumulator_value', -4, 0),
        'epsilon': hp.loguniform('epsilon', -8, -4)
    },
    'adam': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'beta_1': 1 - hp.loguniform('beta_1', -4, 0),
        'beta_2': 1 - hp.loguniform('beta_2', -4, 0),
        'epsilon': hp.loguniform('epsilon', -8, -4)
    },
    'adamax': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'beta_1': 1 - hp.loguniform('beta_1', -4, 0),
        'beta_2': 1 - hp.loguniform('beta_2', -4, 0),
        'epsilon': hp.loguniform('epsilon', -8, -4)
    },
    'momentum': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'momentum': 1 - hp.loguniform('momentum', -4, 0)
    },
    'nadam': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'beta_1': 1 - hp.loguniform('beta_1', -4, 0),
        'beta_2': 1 - hp.loguniform('beta_2', -4, 0),
        'epsilon': hp.loguniform('epsilon', -8, -4)
    },
    'nesterov': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'momentum': 1 - hp.loguniform('momentum', -4, 0)
    },
    'rmsprop': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'rho': 1 - hp.loguniform('rho', -4, 0),
        'epsilon': hp.loguniform('epsilon', -8, -4)
    },
    'sgd': {
        'learning_rate': hp.loguniform('learning_rate', -4, 0)
    }
}
