import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from tarin2 import get_model2

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental_run_functions_eagerly(True)

data_name, feature_name = "all_train_data20random_missing.npz", "train_feature20random_missing.npy"

param_grid = dict(F_=[5,7,9,15,18],
                  gru_output_dim=[7,12,18],
                  conv1d_filters=[7,12,18],
                  conv1d_kernel_size=[3,5,7],
                  conv1d_strides=[1],
                  optimizer_lr=[0.01,0.02,0.007],
                  gat_heads=[1,2],
                  bigru_merge_mode=['sum', 'mul', 'concat', 'ave'])
model = KerasRegressor(build_fn=get_model2)
grid = GridSearchCV(model, param_grid=param_grid, n_jobs=1)

train_feature = np.load('./data/{}'.format(feature_name))
train_speed_data = np.load("./data/{}".format(data_name))['arr_0']
grid_result = grid.fit(X=train_feature, y=train_speed_data)

res_str='best {} using {}'.format(grid_result.best_score_, grid_result.best_params_)
with open('./grid_search_res.txt','a+') as f:
    f.write(res_str)
print(res_str)
