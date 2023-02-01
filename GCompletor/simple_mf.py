import tensorflow as tf
import numpy as np
import os
base_path = './data/seattle/random_missing'
data_path = 'seattle_random_missing20.npy'
data_mat = np.load(os.path.join(base_path, data_path))

