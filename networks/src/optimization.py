import sys
import os
import shutil
import json
import numpy as np
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from space import space
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TerminateOnNaN
from hyperdash import Experiment as HyperdashExperiment
from datasets import Datasets
from models import Models
from optimizers import Optimizers
from callbacks.hyperdash import Hyperdash


class Optimization:
    def __init__(self, dataset_name, model_name, optimizer_name):
        """
        :param dataset_name: name of the dataset
        :type dataset_name: str
        :param model_name: name of the model
        :type model_name: str
        :param optimizer_name: name of the optimizer
        :type optimizer_name: str
        """
        # store names
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.optimizer_name = optimizer_name

        # store search space
        self.space = space[optimizer_name]

        # get config
        with open('../../../networks/src/config.json') as f:
            self.config = json.load(f)

        # get constants
        c = self.config['constants'][dataset_name][model_name]
        self.loss = c['loss']
        self.batch_size = c['batch_size']
        self.epochs = c['epochs']

        # configure and initialize directory
        d = self.main_dir = f'../data/{dataset_name}_{model_name}_{optimizer_name}'
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    def begin(self):
        # optimize hyperparameters
        trials = Trials()
        result = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials,
            verbose=True
        )

        # save log
        with open(f'{self.main_dir}/log.json', 'w') as f:
            json.dump(trials.trials, f, indent=4, default=str)

        # save result
        # subtract from 1 for decay rates
        result = {k: 1 - v if k in ('rho', 'beta_1', 'beta_2', 'momentum') else v for k, v in result.items()}
        with open(f'{self.main_dir}/result.json', 'w') as f:
            json.dump(result, f, indent=4)

    def objective(self, params):
        """
        objective function to optimize

        :param params: hyperparamters for optimizer
        :return: maximum validation accuracy
        :rtype: float
        """
        # get instances
        dataset = Datasets.get(self.dataset_name)
        model = Models.get(self.model_name, dataset=dataset)
        optimizer = Optimizers.get(self.optimizer_name, params=params)

        # configure hyperdash experiment
        hd_exp = HyperdashExperiment(f'{self.dataset_name}', api_key_getter=lambda: self.config['hyperdash']['api_key'])
        hd_exp.param('dataset_name', self.dataset_name)
        hd_exp.param('model_name', self.model_name)
        hd_exp.param('optimizer_name', self.optimizer_name)

        for k, v in params.items():
            hd_exp.param(k, v)

        # set callbacks
        callbacks = [
            Hyperdash(['accuracy', 'loss', 'val_accuracy', 'val_loss'], hd_exp),
            EarlyStopping('val_accuracy', patience=10, min_delta=0.01, verbose=1),
            TerminateOnNaN()
        ]

        # get data
        (x_train, y_train), *_ = dataset.get_batch()

        # start learning
        model.compile(loss=self.loss, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=2
        )

        # stop hyperdash experiment
        hd_exp.end()

        # return maximum validation accuracy
        val_accuracy = np.array(history.history['val_accuracy'])
        return max(val_accuracy) * (-1)
