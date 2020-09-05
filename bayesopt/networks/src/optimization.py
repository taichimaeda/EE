import os
import shutil
import json
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from loader import Datasets
from loader import Models
from loader import Optimizers
from loader import Hyperdash
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TerminateOnNaN
from hyperdash import Experiment as HyperdashExperiment


class Optimization:
    """ class for handling bayesian optimization """
    def __init__(self, dataset_name, model_name, optimizer_name):
        """
        :param dataset_name: name of the dataset
        :type dataset_name: str
        :param optimizer_name: name of the optimizer
        :type optimizer_name: str
        :param model_name: name of the model
        :type model_name: str
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.optimizer_name = optimizer_name

        # get pbounds
        with open('../../pbounds.json') as f:
            self.pbounds = json.load(f)[optimizer_name]

        # configure and initialize directory
        d = self.main_dir = f'../data/{dataset_name}_{model_name}_{optimizer_name}/'
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    def begin(self):
        """ begin bayesian optimization """
        bayesopt = BayesianOptimization(f=self.func, pbounds=self.pbounds)
        logger = JSONLogger(path=f'{self.main_dir}/log.json')
        bayesopt.subscribe(Events.OPTIMIZATION_STEP, logger)
        bayesopt.maximize(init_points=20, n_iter=20, acq='ucb')

        # save results
        with open(f'{self.main_dir}/result.json', 'w') as f:
            json.dump({'bayesopt.res': bayesopt.res, 'bayesopt.max': bayesopt.max}, f, indent=4)

    def func(self, **params):
        """
        black box function to optimize

        :param params: hyperparamters for optimizer
        :return: return value of `experiment.begin()`
        :rtype: float
        """
        with open('../../../networks/src/constants.json') as f:
            constants = json.load(f)[self.dataset_name][self.model_name]

        loss = constants['loss']
        batch_size = constants['batch_size']
        epochs = constants['epochs']

        dataset = Datasets.get(self.dataset_name)
        model = Models.get(self.model_name, dataset=dataset)
        optimizer = Optimizers.get(self.optimizer_name, params=params)

        # create and configure hyperdash experiment
        def get_api_key():
            with open('../../config.json', 'r') as f:
                config = json.load(f)
                return config['hyperdash']['api_key']

        hd_exp = HyperdashExperiment(f'{self.dataset_name}', api_key_getter=get_api_key)
        hd_exp.param('dataset_name', self.dataset_name)
        hd_exp.param('model_name', self.model_name)
        hd_exp.param('optimizer_name', self.optimizer_name)
        for k, v in params.items():
            hd_exp.param(k, v)

        # set callbacks
        callbacks = [
            Hyperdash(['val_loss', 'loss', 'val_accuracy', 'accuracy'], hd_exp),
            EarlyStopping('val_loss', min_delta=0.1, patience=epochs // 10, verbose=1),
            TerminateOnNaN()
        ]

        # get data and compile model
        (x_train, y_train), *_ = dataset.get_batch()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # start learning
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=2
        )

        # stop hyperdash experiment
        hd_exp.end()

        # return min val_loss for bayesian optimization
        # multiply by negative 1 for maximazing with bayesian optimization
        ret = np.array(history.history['val_loss'])
        ret[np.isnan(ret)] = 1e10
        return min(ret) * (-1.0)
