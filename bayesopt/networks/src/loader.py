"""
module for importing all the modules and pacakges from parant directories

note modules and pacakges in the current directory are no longer available
if their names are identical to those imported here
"""
import os
import sys

current_dir = os.path.dirname(__file__)
target_dir = os.path.join(os.path.dirname(__file__), '../../../networks/src')

sys.path.remove(current_dir)
sys.path.append(target_dir)

from datasets import Datasets
from models import Models
from optimizers import Optimizers
from callbacks.hyperdash import Hyperdash

sys.path.remove(target_dir)
sys.path.append(current_dir)
