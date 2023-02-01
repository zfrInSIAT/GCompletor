import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from baselines.metrics import mape, rmse


class AutoEncoderLayer():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()

    def build(self):
        self.input = Input(shape=(None, self.input_dim), batch_size=1)
        self.encode_layer = Dense(self.output_dim, use_bias=False, activation='relu')
        self.encoded = self.encode_layer(self.input)
        self.encoder = Model(self.input, self.encoded)
        self.decode_layer = Dense(self.input_dim, use_bias=False)
        self.decoded = self.decode_layer(self.encoded)
        self.autoencoder = Model(self.input, self.decoded)


# 构建堆叠DAE
class MultiHeadAutoEncoder():
    def __init__(self, head_nums, io_put_dim):
        self.io_put_dim = io_put_dim
        self.head_nums = head_nums
        self.build()

    def build(self):
        self.input = tf.keras.layers.Input(shape=(None, self.io_put_dim[0]))

        res = []
        for i in range(self.head_nums):
            ae = AutoEncoderLayer(*self.io_put_dim)
            out = ae.autoencoder(self.input)
            res.append(out)
        output = tf.reduce_mean(tf.stack(res, axis=-1), axis=-1)
        self.model = Model(self.input, output)


def first_layer_loss(y_true, y_pred):
    mask = y_true != 0
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.math.reduce_mean(
        tf.math.square(tf.math.subtract(tf.where(mask, y_pred, 0), tf.where(mask, y_true, 0))),
        axis=-1)


def train_layers(sae, whole_train_epochs=None, whole_train_lr=0.01,
                 whole_train_batch_size=None, train_data=None):
    opt = Adam(learning_rate=whole_train_lr)
    sae.model.compile(optimizer=opt,
                      loss=first_layer_loss)  # loss='mean_squared_error')#loss='mean_absolute_error')#
    sae.model.fit(x=train_data, y=train_data, epochs=whole_train_epochs, batch_size=whole_train_batch_size)


if __name__ == '__main__':
    base_path = './data/seattle/random_missing'
    data_path = 'seattle_random_missing20.npy'
    data_mat = np.load(os.path.join(base_path, data_path))
    # data_ten = np.reshape(data_mat, [365, 288, 323])

    sae = MultiHeadAutoEncoder(5, [323, 80])
    sae.model.summary()

    whole_train_epochs = 1200

    whole_train_lr = 0.01

    train_layers(sae, whole_train_epochs=whole_train_epochs,
                 whole_train_lr=whole_train_lr,
                 whole_train_batch_size=1, train_data=tf.expand_dims(data_mat, 0))
    pret_data = sae.model.predict(tf.expand_dims(data_mat, 0))
    # true_data = np.array(pd.read_csv('./data/seattle/Speed2.csv').values).reshape([365, 288, 323])
    true_data = np.array(pd.read_csv('./data/seattle/Speed2.csv').values)
    # mask = data_ten != 0
    pret_data = np.squeeze(pret_data)
    # 计算缺失位置的补全效果，
    mask = data_mat == 0
    sae_res2_mape2 = mape(true_data[mask], pret_data[mask])
    sae_res2_rmse2 = rmse(true_data[mask], pret_data[mask])
    ret_str = "计算缺失位置的补全效果\n"
    ret_str += "sae_data:{},whole_train_epochs:{},whole_train_lr:{},mape={}\n".format(
        data_path, whole_train_epochs, whole_train_lr, sae_res2_mape2)
    ret_str += "sae_data:{},whole_train_epochs:{},whole_train_lr:{},rmse={}\n".format(
        data_path, whole_train_epochs,whole_train_lr, sae_res2_rmse2)
    # 查看非缺失位置的接近程度
    mask = data_mat != 0
    sae_res2_mape2 = mape(true_data[mask], pret_data[mask])
    sae_res2_rmse2 = rmse(true_data[mask], pret_data[mask])
    ret_str += "查看非缺失位置的接近程度\n"
    ret_str += "sae_data:{},whole_train_epochs:{},whole_train_lr:{},mape={}\n".format(
        data_path, whole_train_epochs, whole_train_lr, sae_res2_mape2)
    ret_str += "sae_data:{},whole_train_epochs:{},whole_train_lr:{},rmse={}\n".format(
        data_path, whole_train_epochs, whole_train_lr, sae_res2_rmse2)
    print(ret_str)

    # 计算缺失位置的补全效果
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,mape=15.533595850767082
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,rmse=8.132987553239657
    # 查看非缺失位置的接近程度
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,mape=14.247906747538345
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,rmse=7.134550108042455

    # 每个自编码模块都  增加激活函数
    # 计算缺失位置的补全效果
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,mape=14.421848935450122
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,rmse=7.364562862574893
    # 查看非缺失位置的接近程度
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,mape=14.083423845725163
    # sae_data:seattle_random_missing20.npy,whole_train_epochs:600,whole_train_lr:0.02,rmse=7.07797003700423
