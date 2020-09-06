"""
entry point module for single trial experiments

this accepts the following system arguments:

benchmark_name: name of the benchmark
optimizer_name: name of the optimizer
"""
import sys
from loader import Benchmarks
from loader import Optimizers
from optimization import Optimization


if __name__ == '__main__':
    # check command line args
    if not len(sys.argv) == 3:
        raise Exception('invalid numbers of required variables')
    else:
        # get commnad args
        benchmark_name = sys.argv[1]
        optimizer_name = sys.argv[2]
        if False in (Benchmarks.exists(benchmark_name), Optimizers.exists(optimizer_name)):
            raise Exception('invalid values of required variables')

    # begin optimization
    optimization = Optimization(benchmark_name, optimizer_name)
    optimization.begin()
