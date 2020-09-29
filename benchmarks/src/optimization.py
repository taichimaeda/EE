import sys
import os
import shutil
import json
import numpy as np
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from space import space
from benchmarks import Benchmarks
from optimizers import Optimizers


class Optimization:
    def __init__(self, benchmark_name, optimizer_name):
        """
        :param benchmark_name: name of the benchmark
        :type benchmark_name: str
        :param optimizer_name: name of the optimizer
        :type optimizer_name: str
        """
        # store names
        self.benchmark_name = benchmark_name
        self.optimizer_name = optimizer_name

        # get search space
        self.space = space[optimizer_name]

        # configure and initialize directory
        d = self.main_dir = f'../data/{self.benchmark_name}_{self.optimizer_name}'
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

        # ignore overflow and nan warnings
        np.seterr(all='ignore')

    def begin(self):
        # start optimizing
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
        result = {k: 1 - v if k in ('rho', 'initial_accumulator_value', 'beta_1', 'beta_2', 'momentum') else v for k, v in result.items()}
        with open(f'{self.main_dir}/result.json', 'w') as f:
            json.dump(result, f, indent=4)

    def objective(self, params):
        """
        objective function to optimize

        :param params: hyperparamters for optimizer
        :return: return value of `experiment.begin()`
        :rtype: float
        """
        # get instances
        benchmark = Benchmarks.get(self.benchmark_name)
        optimizer = Optimizers.get(self.optimizer_name, benchmark=benchmark, params=params)

        # initialize coordinates
        # random seed is set to 0
        np.random.seed(0)
        coords = np.array([np.random.rand(100).astype(np.float) * 10 - 5 for _ in range(2)])
        optimum = np.array(benchmark.optimum).reshape(2, 1)

        # update coordinates
        dists_mean_min = np.inf
        wait = 0
        patience = 10
        for i in range(10000):
            coords = optimizer.update(coords)
            if i % 100 == 0:
                dists = (np.sum(coords - optimum, axis=0) ** 2.0) ** 0.5
                # terminate on nan
                if np.any(np.isnan(dists)):
                    break
                # early stopping
                if np.mean(dists) > dists_mean_min:
                    wait += 1
                    if wait > patience:
                        break
                else:
                    wait = 0
                    dists_mean_min = np.mean(dists)

        # return minimum distance in log 10
        return np.log10(dists_mean_min)
