'''
堆叠降噪自编码器(Stacked Denoising AutoEncoder)
编解码各2层，且维数基于中心对称
各层维数：784->256->64->256->784
逐个预训练4个autoencoder：使用前一个训练好的ae的encoder输出作为下一个ae的输入来训练当前ae
堆叠ae:取前面预训练好的4个ae的encoder层前后连接起来，形成最终stacked ae的模型结构
堆叠ae的训练：使用预训练好的参数初始化stacked ae，然后进行全局训练优化
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from metrics import mape,rmse
import numpy as np
import matplotlib.pyplot as plt
import os
# 指定gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##### 设置网络参数 #####
epochs_layer = 100
epochs_whole = 200
batch_size = 1 # 256
origin_dim = 4248*288
h_dim1 = 256
h_dim2 = 64


true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature=np.load('../feature.npz').values()

speed_feature=np.squeeze(speed_feature,axis=-1).reshape(1,-1)     # 把三维（288，4248，1）最后一个维度去掉
train_speed_feature = np.squeeze(train_speed_feature,axis=-1).reshape((1,-1))
print(speed_feature.shape)
mask = true_mask ^ train_mask       # 两个掩码相异或，得到的才是在训练中没有，但我们又能得到的数据。

mask = np.squeeze(mask,-1).reshape((1,-1))
train_mask=np.squeeze(train_mask,-1).reshape((1,-1))


assert len(speed_feature.shape) == 2 and len(train_speed_feature.shape)==2 and len(mask.shape)==2

##### 准备mnist数据 ######
# (x_train, _), (x_test, _) = mnist.load_data(path='mnist.npz')
# x_train = x_train.astype('float32')/255.
# x_test = x_test.astype('float32')/255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# x_train = np.expand_dims(train_speed_feature,axis=0)
x_train = train_speed_feature
# 给数据添加噪声
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_train.shape)
# x_test_noisy = x_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

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

loss_masked = Loss_masked(train_mask)

##### 构建单个autoencoder #####
class AutoEncoderLayer():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()

    def build(self):
        self.input = Input(shape=(self.input_dim,),batch_size=1)
        self.encode_layer = Dense(self.output_dim, activation='sigmoid')
        self.encoded = self.encode_layer(self.input)
        self.encoder = Model(self.input, self.encoded)

        self.decode_layer = Dense(self.input_dim, activation='sigmoid')
        self.decoded = self.decode_layer(self.encoded)

        self.autoencoder = Model(self.input, self.decoded)


# 构建堆叠DAE
class StackedAutoEncoder():
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.build()

    def build(self):
        out = self.layer_list[0].encoded
        for i in range(1, num_layers - 1):
            out = self.layer_list[i].encode_layer(out)
        self.model = Model(self.layer_list[0].input, out)




def train_layers(encoder_list=None, layer=None, epochs=None, batch_size=None):
    '''
    预训练：逐层训练，当训练第layer个ae时，使用前（layer-1）个ae训练好的encoder的参数
    :param encoder_list:
    :param layer:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # 对前(layer-1)层用已经训练好的参数进行前向计算，ps:第0层没有前置层
    out = x_train_noisy
    origin = x_train
    if layer != 0:
        for i in range(layer):
            # print("encoder weight", str(i), ":", encoder_list[i].encoder.get_weights()[0])
            out = encoder_list[i].encoder.predict(out)
    if layer ==0:
        encoder_list[layer].autoencoder.summary()
        optimizer = Adam(learning_rate=0.05)
        encoder_list[layer].autoencoder.compile(optimizer=optimizer, loss=loss_masked) # 'binary_crossentropy'

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


def train_whole(sae=None, epochs=None, batch_size=None):
    '''
    用预训练好的参数初始化stacked ae的参数，然后进行全局训练优化
    :param model:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # print("stacked sae weights:")
    # print(sae.model.get_weights())
    sae.model.summary()
    optimizer = Adam(learning_rate=0.05)
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


# 5层的stacked ae，实际上要使用4个ae，实例化4个ae
num_layers = 5
encoder_1 = AutoEncoderLayer(origin_dim, h_dim1)
encoder_2 = AutoEncoderLayer(h_dim1, h_dim2)
decoder_3 = AutoEncoderLayer(h_dim2, h_dim1)
decoder_4 = AutoEncoderLayer(h_dim1, origin_dim)
autoencoder_list = [encoder_1, encoder_2, decoder_3, decoder_4]

# 按照顺序对每一层进行预训练
print("Pre training:")
for level in range(num_layers - 1):
    print("level:", level)
    train_layers(encoder_list=autoencoder_list, layer=level, epochs=epochs_layer, batch_size=batch_size)


# 用训练好的4个ae构建stacked dae
stacked_ae = StackedAutoEncoder(autoencoder_list)
print("Whole training:")
# 进行全局训练优化
train_whole(sae=stacked_ae, epochs=epochs_whole, batch_size=batch_size)

decoder_res = stacked_ae.model.predict(x_train)
stacked_ae_res2_mape2 = mape(speed_feature[mask], decoder_res[mask])
stacked_ae_res2_rmse2 = rmse(speed_feature[mask], decoder_res[mask])
print("stacked_ae mape:{}, rmse:{}".format(stacked_ae_res2_mape2, stacked_ae_res2_rmse2))