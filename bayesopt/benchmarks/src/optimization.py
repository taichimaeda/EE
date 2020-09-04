import os
import shutil
import json
import numpy as np
from decimal import Decimal
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from loader import Benchmarks
from loader import Optimizers


class Optimization:
    """ class for handling bayesian optimization """
    def __init__(self, benchmark_name, optimizer_name):
        """
        :param benchmark_name: name of the benchmark
        :type benchmark_name: str
        :param optimizer_name: name of the optimizer
        :type optimizer_name: str
        """
        self.benchmark_name = benchmark_name
        self.optimizer_name = optimizer_name

        # get pbounds
        with open('../../pbounds.json') as f:
            self.pbounds = json.load(f)[optimizer_name]

        # configure and initialize directory
        d = self.main_dir = f'../data/{self.benchmark_name}_{self.optimizer_name}/'
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
        benchmark = Benchmarks.get(self.benchmark_name)
        optimizer = Optimizers.get(self.optimizer_name, benchmark=benchmark, params=params)

        # set starting coordinates
        # seed is set to 0
        np.random.seed(0)
        coords = [np.random.rand(100).astype(np.float128) * 10 - 5 for _ in range(2)]
        optimum = np.array(benchmark.optimum).reshape(2, 1)
        optimum_dec = np.array([Decimal(str(t)) for t in optimum])

        dists_history = []
        for i in range(10000):
            coords = optimizer.update(coords)
            if i % 100 == 0:
                if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                    break
                # use decimal for precision
                coords_dec = np.array([[Decimal(str(t)) for t in coords[i]] for i in range(2)])
                dists = (np.sum(coords_dec - optimum_dec, axis=0) ** 2.0) ** 0.5
                dists_history.append(dists)

        # return min distance from the optimum solution
        # multiply by negative 1 for maximazing with bayesian optimization
        dists_ave = np.average(dists_history, axis=1)
        return float(min(dists_ave).log10()) * (-1.0)
