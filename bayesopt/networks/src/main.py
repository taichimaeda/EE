"""
entry point module for single trial experiments

this accepts the following system arguments:

dataset_name: name of the dataset
optimizer_name: name of the optimizer
model_name: name of the model
resume: whether to resume from the previous experiment ('true' or 'false' as string)
"""
import sys
from loader import Datasets
from loader import Models
from loader import Optimizers
from optimization import Optimization


if __name__ == '__main__':
    # check command line args
    if not len(sys.argv) == 4:
        raise Exception('invalid numbers of required variables')
    else:
        # get commnad args
        dataset_name = sys.argv[1]
        model_name = sys.argv[2]
        optimizer_name = sys.argv[3]
        if False in (Datasets.exists(dataset_name), Models.exists(model_name), Optimizers.exists(optimizer_name)):
            raise Exception('invalid values of required variables')

    # begin optimization
    optimization = Optimization(dataset_name, model_name, optimizer_name)
    optimization.begin()
