"""
entry point module for multiple trial experiments

this can be execute locally as the process is relatively quick
"""
from experiment import Experiment
from benchmarks import Benchmarks
from optimizers import Optimizers


if __name__ == '__main__':
    # for each benchmark and optimizer
    for benchmark_name in Benchmarks.list_all():
        for optimizer_name in Optimizers.list_all():
            # create and begin experiment
            experiment = Experiment(benchmark_name, optimizer_name)
            experiment.begin()
