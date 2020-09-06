import os
import shutil
import json
import numpy as np
from decimal import Decimal
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from space import space
from loader import Benchmarks
from loader import Optimizers


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

        # get instance
        self.benchmark = Benchmarks.get(self.benchmark_name)

        # get search space
        self.space = space[optimizer_name]

        # configure and initialize directory
        d = self.main_dir = f'../data/{self.benchmark_name}_{self.optimizer_name}/'
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
            max_evals=100,
            trials=trials,
            verbose=True
        )

        # save log
        with open(f'{self.main_dir}/log.json', 'w') as f:
            json.dump(trials.trials, f, indent=4, default=str)

        # save result
        with open(f'{self.main_dir}/result.json', 'w') as f:
            json.dump(result, f, indent=4)

    def objective(self, params):
        """
        objective function to optimize

        :param params: hyperparamters for optimizer
        :return: return value of `experiment.begin()`
        :rtype: float
        """
        # get optimizer instance
        # benchmark instance is obtained in advance since they are indpependent from the given parameters
        optimizer = Optimizers.get(self.optimizer_name, benchmark=self.benchmark, params=params)

        # initialize coordinates
        # random seed is set to 0
        np.random.seed(0)
        coords = [np.random.rand(100).astype(np.float128) * 10 - 5 for _ in range(2)]
        optimum_dec = np.array([Decimal(str(t)) for t in self.benchmark.optimum])
        optimum_dec = optimum_dec.reshape(2, 1)

        # update coordinates
        dists_history = []
        for i in range(10000):
            coords = optimizer.update(coords)
            if i % 100 == 0:
                # use decimal for precision
                coords_dec = np.array([[Decimal(str(t)) for t in coords[i]] for i in range(2)])
                dists = (np.sum(coords_dec - optimum_dec, axis=0) ** Decimal('2.0')) ** Decimal('0.5')
                dists_history.append(dists)

        # return min distance from the optimum solution
        ret = np.average(dists_history, axis=1)
        ret = [Decimal('1e10') if t.is_nan() else t for t in ret]
        return float(min(ret).log10())
