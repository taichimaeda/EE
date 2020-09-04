"""
entry point module for multiple trial experiments

this does not accept any system arguments as the experiment process
completes relatively quickly
"""
import json
from experiment import Experiment
from benchmarks import Benchmarks
from optimizers import Optimizers


if __name__ == '__main__':
    # for each benchmark and optimizer
    for benchmark_name in Benchmarks.list_all():
        for optimizer_name in Optimizers.list_all():
            # get optimized hyperparameters
            with open(f'../../bayesopt/benchmarks/data/{benchmark_name}_{optimizer_name}/result.json') as f:
                params = json.load(f)['bayesopt.max']['params']

            # create and begin experiment
            experiment = Experiment(benchmark_name, optimizer_name, params=params)
            experiment.begin()
