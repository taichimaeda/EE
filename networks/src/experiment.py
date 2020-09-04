import os
import shutil
import json
from datasets import Datasets
from models import Models
from optimizers import Optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
from callbacks.hyperdash import Hyperdash
from callbacks.timelogger import TimeLogger
from hyperdash import Experiment as HyperdashExperiment
from logger import Logger


# create logger
logger = Logger()


class Experiment:
    """ class for handling experiment """
    @logger.read
    def __init__(self, dataset_name, model_name, optimizer_name, trial_num, params):
        """
        :param dataset_name: name of the dataset
        :type dataset_name: str
        :param optimizer_name: name of the optimizer
        :type optimizer_name: str
        :param model_name: name of the model
        :type model_name: str
        :param trial_num: current number of repeated trials
        :type trial_num: int
        """
        with open('./constants.json') as f:
            constants = json.load(f)[dataset_name][model_name]

        self.loss = constants['loss']
        self.batch_size = constants['batch_size']
        self.epochs = constants['epochs']

        self.dataset = Datasets.get(dataset_name)
        self.model = Models.get(model_name, dataset=self.dataset)
        self.optimizer = Optimizers.get(optimizer_name, params=params)

        # configure and initialize directory
        d = self.main_dir = f'../data/{dataset_name}_{model_name}_{optimizer_name}/trial{trial_num}'
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

        # create and configure hyperdash experiment
        def get_api_key():
            with open('../../config.json', 'r') as f:
                config = json.load(f)
                return config['hyperdash']['api_key']

        self.hd_exp = HyperdashExperiment(f'{dataset_name}', api_key_getter=get_api_key)
        self.hd_exp.param('dataset_name', dataset_name)
        self.hd_exp.param('model_name', model_name)
        self.hd_exp.param('optimizer_name', optimizer_name)
        self.hd_exp.param('trial_num', trial_num)
        for k, v in params.items():
            self.hd_exp.param(k, v)

        # set callbacks
        self.callbacks = [
            Hyperdash(['val_loss', 'loss', 'val_accuracy', 'accuracy'], self.hd_exp),
            TensorBoard(log_dir=f'{self.main_dir}/tensorboard'),
            TimeLogger(filename=f'{self.main_dir}/time.csv'),
            CSVLogger(filename=f'{self.main_dir}/result.csv', append=True)
        ]

    @logger.write
    def begin(self):
        """ begin experiment """
        try:
            # get data
            (x_train, y_train), (x_test, y_test) = self.dataset.get_batch()
            # start learning
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
            self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                callbacks=self.callbacks,
                verbose=2
            )

            # print final scores
            score = self.model.evaluate(x_test, y_test, verbose=1)
            print('test loss:', score[0])
            print('test accuracy:', score[1])

        # catch keyboard interrupt to stop hyperdash experiment
        except KeyboardInterrupt:
            pass
        finally:
            self.hd_exp.end()
