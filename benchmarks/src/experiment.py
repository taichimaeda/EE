import os
import shutil
import csv
import numpy as np
from decimal import Decimal
from datetime import datetime
from benchmarks import Benchmarks
from optimizers import Optimizers


class Experiment:
    """ class for handling experiment """
    def __init__(self, benchmark_name, optimizer_name, params):
        """
        :param benchmark_name: name of the benchmark
        :type benchmark_name: str
        :param optimizer_name: name of the optimizer
        :type optimizer_name: str
        """
        self.benchmark = Benchmarks.get(benchmark_name)
        self.optimizer = Optimizers.get(optimizer_name, benchmark=self.benchmark, params=params)

        # configure and initialize directory
        d = self.main_dir = f'../data/{benchmark_name}_{optimizer_name}'
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    def begin(self):
        """ begin experiment """
        # set starting coordinates
        # seed is set to 0
        np.random.seed(0)
        coords = np.array([np.random.rand(100).astype(np.float128) * 10 - 5 for _ in range(2)])
        optimum = np.array(self.benchmark.optimum).reshape(2, 1)
        optimum_dec = np.array([Decimal(str(t)) for t in optimum])

        dists_history = []
        time_history = []
        start = datetime.now()
        for i in range(10000):
            coords = self.optimizer.update(coords)
            if i % 100 == 0:
                if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                    break
                coords_dec = np.array([[Decimal(str(t)) for t in coords[i]] for i in range(2)])
                dists = (np.sum(coords_dec - optimum_dec, axis=0) ** 2.0) ** 0.5
                dists_history.append(dists)
                time_history.append((datetime.now() - start).total_seconds() / 100)
                start = datetime.now()

        # save dists history
        with open(f'{self.main_dir}/result.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['step', *[f'#{i + 1}' for i in range(len(dists_history[0]))]]
            writer.writerow(header)
            for i in range(len(dists_history)):
                row = [(i + 1) * 100,
                       *dists_history[i]]
                writer.writerow(row)

        # save time history
        with open(f'{self.main_dir}/time.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['step', 'average time']
            writer.writerow(header)
            for i in range(len(time_history)):
                row = [(i + 1) * 100, time_history[i]]
                writer.writerow(row)
