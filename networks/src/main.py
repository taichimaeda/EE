"""
entry point module for single trial experiments

this accepts the following system arguments:

dataset_name: name of the dataset
optimizer_name: name of the optimizer
model_name: name of the model
trial_num: current number of repeated trials
"""
import sys
import json
from experiment import Experiment
from datasets import Datasets
from models import Models
from optimizers import Optimizers


if __name__ == '__main__':
    # check command line args
    if not len(sys.argv) == 5:
        raise Exception('invalid numbers of required variables')
    else:
        # get commnad args
        dataset_name = sys.argv[1]
        model_name = sys.argv[2]
        optimizer_name = sys.argv[3]
        trial_num = int(sys.argv[4])
        if False in (Datasets.exists(dataset_name), Models.exists(model_name), Optimizers.exists(optimizer_name)) or trial_num >= 3:
            raise Exception('invalid values of required variables')

    # get optimized hyperparameters
    with open(f'../../bayesopt/networks/data/{dataset_name}_{model_name}_{optimizer_name}/result.json') as f:
        params = json.load(f)['bayesopt.max']['params']

    # create and begin experiment
    experiment = Experiment(dataset_name, optimizer_name, model_name, trial_num, params=params)
    experiment.begin()
