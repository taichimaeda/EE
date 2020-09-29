import os
import shutil
import csv
import json
import numpy as np
from datetime import datetime
from benchmarks import Benchmarks
from optimizers import Optimizers
from logger import Logger


# create logger
logger = Logger()


class Experiment:
    @logger.read
    def __init__(self, benchmark_name, optimizer_name):
        """
        :param benchmark_name: name of the benchmark
        :type benchmark_name: str
        :param optimizer_name: name of the optimizer
        :type optimizer_name: str
        """
        # get optimized hyperparameters
        with open(f'../params/{benchmark_name}_{optimizer_name}/result.json') as f:
            params = json.load(f)

        # get instances
        self.benchmark = Benchmarks.get(benchmark_name)
        self.optimizer = Optimizers.get(optimizer_name, benchmark=self.benchmark, params=params)

        # configure and initialize directory
        d = self.main_dir = f'../data/{benchmark_name}_{optimizer_name}'
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    @logger.write
    def begin(self):
        # initialize coordinates
        # random seed is set to 0
        np.random.seed(0)
        coords = np.array([np.random.rand(100).astype(np.float) * 10 - 5 for _ in range(2)])
        optimum = np.array(self.benchmark.optimum).reshape(2, 1)

        # update coordinates
        dists_history = []
        time_history = []
        start = datetime.now()
        for i in range(10000):
            coords = self.optimizer.update(coords)
            if i % 100 == 0:
                dists = (np.sum(coords - optimum, axis=0) ** 2.0) ** 0.5
                dists_history.append(dists)
                time_history.append((datetime.now() - start).total_seconds() / 100)
                start = datetime.now()

        # save dists history
        with open(f'{self.main_dir}/result.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['step', 'average distance']
            writer.writerow(header)
            for i in range(len(dists_history)):
                row = [(i + 1) * 100, np.mean(dists_history[i])]
                writer.writerow(row)

        # save time history
        with open(f'{self.main_dir}/time.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['step', 'average time']
            writer.writerow(header)
            for i in range(len(time_history)):
                row = [(i + 1) * 100, time_history[i]]
                writer.writerow(row)
