import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from baselines.metrics import mape, rmse
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler


class Loss_masked(tf.keras.losses.Loss):
    def __init__(self, mask, **kwargs):
        self.mask = tf.expand_dims(mask, 0)
        super(Loss_masked, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        # print(self.mask.shape,y_true.shape,y_pred.shape)
        # assert self.mask.shape==y_true.shape==y_pred.shape
        # y_pred = tf.convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.math.reduce_mean(
            tf.math.square(tf.math.subtract(tf.where(self.mask, y_pred, 0), tf.where(self.mask, y_true, 0))), axis=-1)


class AutoEncoderLayer():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()

    def build(self):
        self.input = Input(shape=(self.input_dim,), batch_size=1)
        self.encode_layer = Dense(self.output_dim, activation='sigmoid')
        self.encoded = self.encode_layer(self.input)
        self.encoder = Model(self.input, self.encoded)

        self.decode_layer = Dense(self.input_dim, activation='sigmoid')
        self.decoded = self.decode_layer(self.encoded)

        self.autoencoder = Model(self.input, self.decoded)


class StackedAutoEncoder():
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.num_layers = len(layer_list)+1
        self.build()

    def build(self):
        out = self.layer_list[0].encoded
        for i in range(1, self.num_layers - 1):
            out = self.layer_list[i].encode_layer(out)
        self.model = Model(self.layer_list[0].input, out)


def train_layers(x_train, encoder_list=None, layer=None, epochs=None, batch_size=None, learning_rate=0.05):
    '''
    预训练：逐层训练，当训练第layer个ae时，使用前（layer-1）个ae训练好的encoder的参数
    :param encoder_list:
    :param layer:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # 对前(layer-1)层用已经训练好的参数进行前向计算，ps:第0层没有前置层
    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    # x_test_noisy = x_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    out = x_train_noisy
    origin = x_train
    if layer != 0:
        for i in range(layer):
            # print("encoder weight", str(i), ":", encoder_list[i].encoder.get_weights()[0])
            out = encoder_list[i].encoder.predict(out)
    if layer == 0:
        encoder_list[layer].autoencoder.summary()
        optimizer = Adam(learning_rate)
        encoder_list[layer].autoencoder.compile(optimizer=optimizer,
                                                loss='binary_crossentropy')  # 'binary_crossentropy'

        # 训练第layer个ae
        encoder_list[layer].autoencoder.fit(
            out,
            origin if layer == 0 else out,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=2
        )
    else:
        encoder_list[layer].autoencoder.summary()
        encoder_list[layer].autoencoder.compile(optimizer='adadelta',
                                                loss='mean_absolute_error')  # 'binary_crossentropy'

        # 训练第layer个ae
        encoder_list[layer].autoencoder.fit(
            out,
            origin if layer == 0 else out,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=2
        )


def train_whole(x_train, sae=None, epochs=None, batch_size=None,learning_rate=0.05):
    '''
    用预训练好的参数初始化stacked ae的参数，然后进行全局训练优化
    :param model:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # print("stacked sae weights:")
    # print(sae.model.get_weights())

    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    # x_test_noisy = x_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    # sae.model.summary()
    optimizer = Adam(learning_rate)
    sae.model.compile(optimizer=optimizer, loss='mean_absolute_error')  # 'binary_crossentropy'
    sae.model.fit(
        x_train_noisy,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        # shuffle=True,
        # validation_data=(x_test_noisy, x_test),
        verbose=2
    )


def sae(x_train,
        epochs_layer=100,
        epochs_whole=200,
        batch_size = 1,
        pre_learning_rate=0.05,
        whole_learning_rate=0.05
        ):
      # 256
    origin_dim = x_train.shape[-1]
    h_dim1 = 256
    h_dim2 = 64
    # 5层的stacked ae，实际上要使用4个ae，实例化4个ae
    num_layers = 4
    encoder_1 = AutoEncoderLayer(origin_dim, h_dim1)
    encoder_2 = AutoEncoderLayer(h_dim1, h_dim2)
    decoder_3 = AutoEncoderLayer(h_dim2, h_dim1)
    decoder_4 = AutoEncoderLayer(h_dim1, origin_dim)
    autoencoder_list = [encoder_1, encoder_2, decoder_3, decoder_4]

    # 按照顺序对每一层进行预训练
    print("Pre training:")
    for level in range(num_layers - 1):
        print("level:", level)
        train_layers(x_train=x_train,encoder_list=autoencoder_list, layer=level, epochs=epochs_layer, batch_size=batch_size,learning_rate=pre_learning_rate)

    # 用训练好的4个ae构建stacked dae
    stacked_ae = StackedAutoEncoder(autoencoder_list)
    print("Whole training:")
    # 进行全局训练优化
    train_whole(x_train=x_train,sae=stacked_ae, epochs=epochs_whole, batch_size=batch_size,learning_rate=whole_learning_rate)

    decoder_res = stacked_ae.model.predict(x_train)
    return decoder_res
    # stacked_ae_res2_mape2 = mape(speed_feature[mask], decoder_res[mask])
    # stacked_ae_res2_rmse2 = rmse(speed_feature[mask], decoder_res[mask])
    # print("stacked_ae mape:{}, rmse:{}".format(stacked_ae_res2_mape2, stacked_ae_res2_rmse2))
