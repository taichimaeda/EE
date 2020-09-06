from hyperopt import hp
import numpy as np


space = {
    'adadelta': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'rho': 1 - hp.loguniform('rho', np.log(1e-4), np.log(1.0)),
        'epsilon': hp.loguniform('epsilon', np.log(1e-8), np.log(1e-4))
    },
    'adagrad': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'initial_accumulator_value': 1 - hp.loguniform('initial_accumulator_value', np.log(1e-4), np.log(1.0)),
        'epsilon': hp.loguniform('epsilon', np.log(1e-8), np.log(1e-4))
    },
    'adam': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'beta_1': 1 - hp.loguniform('beta_1', np.log(1e-4), np.log(1.0)),
        'beta_2': 1 - hp.loguniform('beta_2', np.log(1e-4), np.log(1.0)),
        'epsilon': hp.loguniform('epsilon', np.log(1e-8), np.log(1e-4))
    },
    'adamax': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'beta_1': 1 - hp.loguniform('beta_1', np.log(1e-4), np.log(1.0)),
        'beta_2': 1 - hp.loguniform('beta_2', np.log(1e-4), np.log(1.0)),
        'epsilon': hp.loguniform('epsilon', np.log(1e-8), np.log(1e-4))
    },
    'momentum': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'momentum': 1 - hp.loguniform('momentum', np.log(1e-4), np.log(1.0))
    },
    'nadam': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'beta_1': 1 - hp.loguniform('beta_1', np.log(1e-4), np.log(1.0)),
        'beta_2': 1 - hp.loguniform('beta_2', np.log(1e-4), np.log(1.0)),
        'epsilon': hp.loguniform('epsilon', np.log(1e-8), np.log(1e-4))
    },
    'nesterov': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'momentum': 1 - hp.loguniform('momentum', np.log(1e-4), np.log(1.0))
    },
    'rmsprop': {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(10.0)),
        'rho': 1 - hp.loguniform('rho', np.log(1e-4), np.log(1.0)),
        'epsilon': hp.loguniform('epsilon', np.log(1e-8), np.log(1e-4))
    },
    'sgd': {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(10.0))
    }
}
