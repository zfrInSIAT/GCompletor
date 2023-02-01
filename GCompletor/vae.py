#! -*- coding: utf-8 -*-

'''用Keras实现的VAE
   目前只保证支持Tensorflow后端
   改写自
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from scipy.stats import norm
import sys
sys.path.append(r'./baselines')
from baselines.metrics import mape, rmse
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist


batch_size = 800
original_dim = 14*288
latent_dim = 2 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 512
epochs = 15

true_all_data = np.load('./all_speed_data.npz')
true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
# 保存的shape为（14，288，4248,1），需要squeeze一下
true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature), np.squeeze(true_all_mask)

# 以下shape均为（14，288，4248）, 随机缺失
data20miss = np.load("./data/all_train_data20random_missing.npz")
train_speed_data20miss, train_mask20miss = data20miss['arr_0'], data20miss['arr_1']
data40miss = np.load("./data/all_train_data40random_missing.npz")
train_speed_data40miss, train_mask40miss = data40miss['arr_0'], data40miss['arr_1']
data60miss = np.load("./data/all_train_data60random_missing.npz")
train_speed_data60miss, train_mask60miss = data60miss['arr_0'], data60miss['arr_1']
data80miss = np.load("./data/all_train_data80random_missing.npz")
train_speed_data80miss, train_mask80miss = data80miss['arr_0'], data80miss['arr_1']
true_all_speed_feature = true_all_speed_feature.transpose([2, 0, 1])
true_all_mask = true_all_mask.transpose([2, 0, 1])
train_speed_data20miss = train_speed_data20miss.transpose([2, 0, 1])
train_speed_data40miss = train_speed_data40miss.transpose([2, 0, 1])
train_speed_data60miss = train_speed_data60miss.transpose([2, 0, 1])
train_speed_data80miss = train_speed_data80miss.transpose([2, 0, 1])
train_mask20miss = train_mask20miss.transpose([2, 0, 1])
train_mask40miss = train_mask40miss.transpose([2, 0, 1])
train_mask60miss = train_mask60miss.transpose([2, 0, 1])
train_mask80miss = train_mask80miss.transpose([2, 0, 1])
# 操作完之后的shape为（4248，14，288）

mask20_ten = true_all_mask ^ train_mask20miss
mask40_ten = true_all_mask ^ train_mask40miss
mask60_ten = true_all_mask ^ train_mask60miss
mask80_ten = true_all_mask ^ train_mask80miss

# 感觉这样直接转换会出问题，在其他代码中，是行为路段，列为时间序列
# 需要先换轴，在reshape
# 转化后的shape为（4248,14*288)
# 注意前面已经transpose过了
true_all_speed_feature_mat = true_all_speed_feature.reshape(4248, -1)
true_all_mask_mat = true_all_mask.reshape(4248, -1)
train_speed_data20miss_mat = train_speed_data20miss.reshape(4248, -1)
scalar = StandardScaler()
train_speed_data20miss_mat = scalar.fit_transform(train_speed_data20miss_mat)
train_mask20miss_mat = train_mask20miss.reshape(4248,-1)
train_speed_data40miss_mat = train_speed_data40miss.reshape(4248,-1)
train_mask40miss_mat = train_mask40miss.reshape(4248,-1)
train_speed_data60miss_mat = train_speed_data60miss.reshape(4248, -1)
train_mask60miss_mat = train_mask60miss.reshape(4248, -1)
train_speed_data80miss_mat = train_speed_data80miss.reshape(4248,-1)
train_mask80miss_mat = train_mask80miss.reshape(4248, -1)
mask20_mat = train_mask20miss_mat ^ true_all_mask_mat
mask40_mat = train_mask40miss_mat ^ true_all_mask_mat
mask60_mat = train_mask60miss_mat ^ true_all_mask_mat
mask80_mat = train_mask80miss_mat ^ true_all_mask_mat
# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# 建立模型
vae = Model(x, x_decoded_mean)

# xent_loss是重构loss，kl_loss是KL loss

xent_loss = K.sum(K.square(tensorflow.math.subtract(x, x_decoded_mean)), axis=-1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss*0.01 + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01))
vae.summary()

vae.fit(true_all_speed_feature_mat[:3248],
        shuffle=True,
        epochs=30,
        batch_size=batch_size,)


# 构建encoder，然后观察各个数字在隐空间的分布
encoder = Model(x, z_mean)

# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
# plt.colorbar()
# plt.show()

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# 观察隐变量的两个维度变化是如何影响输出结果的
n = 5  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

#用正态分布的分位数来构建隐变量对
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

res = ""
x_decoded = vae.predict(train_speed_data20miss_mat[3248:])
x_decoded = scalar.inverse_transform(x_decoded)
BTRMF_res2_mape2 = mape(true_all_speed_feature_mat[3248:][mask20_mat[3248:]], x_decoded[mask20_mat[3248:]])
BTRMF_res2_rmse2 = rmse(true_all_speed_feature_mat[3248:][mask20_mat[3248:]], x_decoded[mask20_mat[3248:]])
res += "BTRMF_missing_rate,rank=,mape={},rmse={}\n".format( BTRMF_res2_mape2, BTRMF_res2_rmse2)
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = generator.predict(z_sample)
#         BTRMF_res2_mape2 = mape(true_all_speed_feature_mat[mask20_mat], x_decoded[mask20_mat])
#         BTRMF_res2_rmse2 = rmse(true_all_speed_feature_mat[mask20_mat], x_decoded[mask20_mat])
#         res += "BTRMF_missing_rate,rank=,mape={},rmse={}\n".format( BTRMF_res2_mape2, BTRMF_res2_rmse2)
        # digit = x_decoded[0].reshape(digit_size, digit_size)
        # figure[i * digit_size: (i + 1) * digit_size,
        #        j * digit_size: (j + 1) * digit_size] = digit
print(res)
# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()