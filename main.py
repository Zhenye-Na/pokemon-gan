"""High level pipeline for Pokemon WGAN."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

# from models.support_vector_machine import SupportVectorMachine
# from train_eval_model import train_model, eval_model, train_model_qp
from utils.io_tools import read_dataset
# from utils.data_tools import preprocess_data



# Read image into dictionary
data = train_set = read_dataset("data/train.txt", "data/image_data/")

# Preprocess all of the Pokemon images
preprocess_data("./data/image_data", "./data/preprocessed_data", 'default')