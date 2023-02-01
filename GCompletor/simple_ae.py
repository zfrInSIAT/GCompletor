import os.path

import numpy as np
import pandas as pd
# import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from utils import get_deviation
import os
from baselines.metrics import mape, rmse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 用gpu跑内存不够，用cpu可以跑
# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental_run_functions_eagerly(True)

class AutoEncoderLayer():
    def __init__(self,  input_dim, output_dim):
        # self.road = road
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()

    def build(self):
        self.input = Input(shape=(self.input_dim,))
        self.encode_layer = Dense(self.output_dim, use_bias=False)
        self.encoded = self.encode_layer(self.input)
        self.encoder = Model(self.input, self.encoded)
        self.decode_layer = Dense(self.input_dim, use_bias=False)
        self.decoded = self.decode_layer(self.encoded)
        self.autoencoder = Model(self.input, self.decoded)


# 构建堆叠DAE
class StackedAutoEncoder():
    def __init__(self, layer_nums, io_put_dim):
        self.io_put_dim = io_put_dim
        self.layer_nums = layer_nums
        # self.road = road
        self.build()

    def build(self):
        self.layers = []
        for i in range(self.layer_nums):
            ae = AutoEncoderLayer(*self.io_put_dim[i])
            self.layers.append(ae)
        out = self.layers[0].encoded
        for i in range(1, self.layer_nums):
            out = self.layers[i].encode_layer(out)
        for i in range(self.layer_nums - 1, -1, -1):
            out = self.layers[i].decode_layer(out)
        self.model = Model(self.layers[0].input, out)


def first_layer_loss(y_true, y_pred):
    mask = y_true != 0
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.math.reduce_mean(
        tf.math.square(tf.math.subtract(tf.where(mask, y_pred, 0), tf.where(mask, y_true, 0))),
        axis=None)


def train_layers(sae, per_train_epochs=None, whole_train_epochs=None, pre_train_lr=0.03, whole_train_lr=0.01,
                 pre_train_batch_size=None, whole_train_batch_size=None, train_data=None):
    # out = train_data
    # 每层分别训练
    train_data = tf.keras.layers.Dropout(rate=0.2)(train_data, training=True)
    # sae_data:missing rate:20,mape=29.560926538372307
    # sae_data:,missing_rate:20,rmse=2.4967354187326483

    # sae_data:missing rate:20,mape=45.738799552122615
    # sae_data:,missing_rate:20,rmse=3.7091217779248455
    # dropout 0.2
    # sae_data:missing rate:20,mape=34.21313589177886
    # sae_data:,missing_rate:20,rmse=2.6766892739570203
    # sae_data:missing rate:20,mape=34.659746360789626
    # sae_data:,missing_rate:20,rmse=2.703192269046301
    # dropout 0
    # sae_data:missing rate:20,mape=33.31666977134457
    # sae_data:,missing_rate:20,rmse=2.735428426626868
    # dropout 0.5
    # sae_data:missing rate:20,mape=34.8870516700724
    # sae_data:,missing_rate:20,rmse=2.6966871477191003
    for layer in range(sae.layer_nums):
        # out = np.multiply(train_data[np.random.choice(6, pre_train_batch_size, replace=False)],
        #                   np.where(np.random.random((pre_train_batch_size, 4248 * 288)) < 0.4, 0., 1.))
        out = train_data
        if layer != 0:
            for i in range(layer):
                # print("encoder weight", str(i), ":", encoder_list[i].encoder.get_weights()[0])
                out = sae.layers[i].encoder.predict(out)
        if layer == 0:
            print("layers,",layer)
            optimizer = Adam(learning_rate=pre_train_lr)
            sae.layers[layer].autoencoder.compile(optimizer=optimizer,
                                                  loss=first_layer_loss)#loss='mean_absolute_error')#loss=first_layer_loss)  #
            # 训练第layer个ae
            sae.layers[layer].autoencoder.fit(
                out,
                out,
                epochs=per_train_epochs,
                batch_size=pre_train_batch_size,
                shuffle=True,
                verbose=1
            )
        else:
            print("layers,", layer)
            sae.layers[layer].autoencoder.compile(optimizer=Adam(learning_rate=pre_train_lr),
                                                  loss='mean_squared_error')  # loss='mean_absolute_error')  # 'binary_crossentropy'
            # 训练第layer个ae
            sae.layers[layer].autoencoder.fit(
                out,
                out,
                epochs=per_train_epochs,
                batch_size=pre_train_batch_size,
                shuffle=True,
                verbose=1
            )
    # 全部一起训练
    # whole_train_data = np.multiply(train_data[np.random.choice(6, pre_train_batch_size, replace=False)],
    #                                np.where(np.random.random((pre_train_batch_size, 4248 * 288)) < 0.4, 0., 1.))
    whole_train_data = train_data
    if whole_train_epochs:
        print("train whole")
        opt = Adam(learning_rate=whole_train_lr)
        sae.model.compile(optimizer=opt,
                          loss=first_layer_loss)#loss='mean_squared_error')#loss='mean_absolute_error')  #
        sae.model.fit(x=whole_train_data, y=whole_train_data, shuffle=True, epochs=whole_train_epochs,
                      batch_size=whole_train_batch_size)


def get_mape_rmse(scaler, train_speed_data20miss_mat, sae, true_all_speed_feature_mat, mask20_mat, t_m, missing_rate):
    train_speed_data20miss_mat = scaler.transform(train_speed_data20miss_mat)
    pret_data = sae.model.predict(train_speed_data20miss_mat)
    pret_data = scaler.inverse_transform(pret_data)
    # true_all_speed_feature_mat = scaler.inverse_transform(true_all_speed_feature_mat)
    # true_all_speed_feature_mat = true_all_speed_feature_mat
    mask20_mat2 = mask20_mat ^ t_m
    a = true_all_speed_feature_mat[mask20_mat2]
    b = pret_data[mask20_mat2]
    sae_res2_mape2 = mape(a, b)
    sae_res2_rmse2 = rmse(a, b)
    ret_str = ""
    # ret_str += "计算缺失位置的补全效果，rank：" + str(i) + "\n"
    ret_str += "sae_data:missing rate:{},mape={}\n".format(
        missing_rate, sae_res2_mape2)
    ret_str += "sae_data:,missing_rate:{},rmse={}\n".format(
        missing_rate, sae_res2_rmse2)
    true_distribution, pret_distribution = get_deviation(true_all_speed_feature_mat, t_m, pret_data)
    ret_str += "真实分布：{}\n".format(true_distribution)
    ret_str += "预测分布：{}\n".format(pret_distribution)

    sorted_list = np.load(r'./data/sorted_var_500_road.npy')
    a = np.reshape(true_all_speed_feature_mat, [-1, 288, 500])[..., sorted_list[400:]]
    b = np.reshape(pret_data, [-1, 288, 500])[..., sorted_list[400:]]
    m = np.reshape(mask20_mat2, [-1, 288, 500])[..., sorted_list[400:]]
    BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    c = np.reshape(true_all_speed_feature_mat, [-1, 288, 500])[..., sorted_list[:400]]
    d = np.reshape(pret_data, [-1, 288, 500])[..., sorted_list[:400]]
    e = np.reshape(mask20_mat2, [-1, 288, 500])[..., sorted_list[:400]]
    BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

    print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)
    print(ret_str)

    speed_feature_ten = np.reshape(true_all_speed_feature_mat, [-1, 288, 500])
    data_ten = np.reshape(pret_data, [-1, 288, 500])
    mask_ten = np.reshape(mask20_mat2, [-1, 288, 500])
    str2 = "分时段\n"
    for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
        t1 = mask_ten[:, item[0]:item[1], :]
        t2 = speed_feature_ten[:, item[0]:item[1], :][t1]
        t3 = data_ten[:, item[0]:item[1], :][t1]
        if len(t2) == 0:
            mean_res_mape1 = 99999
            mean_res_rmse1 = 99999
        else:
            mean_res_mape1 = mape(t2, t3)
            mean_res_rmse1 = rmse(t2, t3)
        str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
        # str2 += "mape:{}\n".format(mean_res_mape1)
    print(str2)

    # deviation_true_data, deviation_pret_data = get_deviation(test_true_data_and_mask[0], test_true_data_and_mask[1],
    #                                                          predict_data)
    # print("BTRMF_deviation_pret_data", deviation_pret_data)
    # print("deviation_true_data", deviation_true_data)
    # speed_feature_ten = np.reshape(true_all_speed_feature_mat, [-1, 288, 500])
    # data_ten = np.reshape(pret_data, [-1, 288, 500])
    # mask_ten = np.reshape(mask20_mat2, [-1, 288, 500])
    # str2 = "分时段\n"
    # for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
    #     mean_res_mape1 = mape(speed_feature_ten[:, item[0]:item[1], :][mask_ten[:, item[0]:item[1], :]],
    #                           data_ten[:, item[0]:item[1], :][mask_ten[:, item[0]:item[1], :]])
    #     mean_res_rmse1 = rmse(speed_feature_ten[:, item[0]:item[1], :][mask_ten[:, item[0]:item[1], :]],
    #                           data_ten[:, item[0]:item[1], :][mask_ten[:, item[0]:item[1], :]])
    #     str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
    # print(str2)


if __name__ == '__main__':

    selected500road_list_and_adj_mat = np.load("./data/selected500road_index_and_adj_mat.npz")
    roadsectId_list = selected500road_list_and_adj_mat['arr_0']
    roadsect_adj_mat = selected500road_list_and_adj_mat['arr_1']

    true_all_data = np.load('./all_speed_data.npz')
    true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    # 保存的shape为（14，288，4248,1），需要squeeze一下
    true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature)[:,:,roadsectId_list], np.squeeze(true_all_mask)[:,:,roadsectId_list]

    # 以下shape均为（14，288，4248）, 随机缺失
    # data20miss = np.load("./data/all_train_data20random_missing.npz")
    # train_speed_data20miss, train_mask20miss = data20miss['arr_0'][:,:,roadsectId_list], data20miss['arr_1'][:,:,roadsectId_list]
    # data40miss = np.load("./data/all_train_data40random_missing.npz")
    # train_speed_data40miss, train_mask40miss = data40miss['arr_0'][:,:,roadsectId_list], data40miss['arr_1'][:,:,roadsectId_list]
    # data60miss = np.load("./data/all_train_data60random_missing.npz")
    # train_speed_data60miss, train_mask60miss = data60miss['arr_0'][:,:,roadsectId_list], data60miss['arr_1'][:,:,roadsectId_list]
    # data80miss = np.load("./data/all_train_data80random_missing.npz")
    # train_speed_data80miss, train_mask80miss = data80miss['arr_0'][:,:,roadsectId_list], data80miss['arr_1'][:,:,roadsectId_list]
    #
    # data20nr_miss = np.load("./data/all_train_data20no_random_missing_road.npz")
    # train_speed_data20nr_miss, train_mask20nr_miss = data20nr_miss['arr_0'][:,:,roadsectId_list], data20nr_miss['arr_1'][:,:,roadsectId_list]
    # data40nr_miss = np.load("./data/all_train_data40no_random_missing_road.npz")
    # train_speed_data40nr_miss, train_mask40nr_miss = data40nr_miss['arr_0'][:,:,roadsectId_list], data40nr_miss['arr_1'][:,:,roadsectId_list]
    # data60nr_miss = np.load("./data/all_train_data60no_random_missing_road.npz")
    # train_speed_data60nr_miss, train_mask60nr_miss = data60nr_miss['arr_0'][:,:,roadsectId_list], data60nr_miss['arr_1'][:,:,roadsectId_list]
    # data80nr_miss = np.load("./data/all_train_data80no_random_missing_road.npz")
    # train_speed_data80nr_miss, train_mask80nr_miss = data80nr_miss['arr_0'][:,:,roadsectId_list], data80nr_miss['arr_1'][:,:,roadsectId_list]

    # true_all_speed_feature = true_all_speed_feature.transpose([0, 2, 1])
    # true_all_mask = true_all_mask.transpose([0, 2, 1])
    # train_speed_data20miss = train_speed_data20miss.transpose([0, 2, 1])
    # train_mask20miss = train_mask20miss.transpose([0, 2, 1])

    # true_all_speed_feature = true_all_speed_feature.transpose([2, 0, 1])
    # true_all_mask = true_all_mask.transpose([2, 0, 1])
    # train_speed_data20miss = train_speed_data20miss.transpose([2, 0, 1])
    # train_speed_data40miss = train_speed_data40miss.transpose([2, 0, 1])
    # train_speed_data60miss = train_speed_data60miss.transpose([2, 0, 1])
    # train_speed_data80miss = train_speed_data80miss.transpose([2, 0, 1])
    # train_mask20miss = train_mask20miss.transpose([2, 0, 1])
    # train_mask40miss = train_mask40miss.transpose([2, 0, 1])
    # train_mask60miss = train_mask60miss.transpose([2, 0, 1])
    # train_mask80miss = train_mask80miss.transpose([2, 0, 1])
    # 操作完之后的shape为（4248，14，288）

    # mask20_ten = true_all_mask ^ train_mask20miss
    # mask40_ten = true_all_mask ^ train_mask40miss
    # mask60_ten = true_all_mask ^ train_mask60miss
    # mask80_ten = true_all_mask ^ train_mask80miss

    # 感觉这样直接转换会出问题，在其他代码中，是行为路段，列为时间序列
    # 需要先换轴，在reshape
    # 转化后的shape为（4248,14*288)
    # 注意前面已经transpose过了
    true_all_speed_feature_mat = true_all_speed_feature.reshape(14, -1)
    true_all_mask_mat = true_all_mask.reshape(14, -1)
    # train_speed_data20miss_mat = train_speed_data20miss.reshape(14, -1)
    # train_mask20miss_mat = train_mask20miss.reshape(14, -1)
    # train_speed_data40miss_mat = train_speed_data40miss.reshape(14, -1)
    # train_mask40miss_mat = train_mask40miss.reshape(14, -1)
    # train_speed_data60miss_mat = train_speed_data60miss.reshape(14, -1)
    # train_mask60miss_mat = train_mask60miss.reshape(14, -1)
    # train_speed_data80miss_mat = train_speed_data80miss.reshape(14, -1)
    # train_mask80miss_mat = train_mask80miss.reshape(14, -1)
    # true_all_speed_feature_mat = true_all_speed_feature.reshape(100, -1)
    # true_all_mask_mat = true_all_mask.reshape(100, -1)
    # train_speed_data20miss_mat = train_speed_data20miss.reshape(100, -1)
    # train_mask20miss_mat = train_mask20miss.reshape(100, -1)
    # train_speed_data40miss_mat = train_speed_data40miss.reshape(100, -1)
    # train_mask40miss_mat = train_mask40miss.reshape(100, -1)
    # train_speed_data60miss_mat = train_speed_data60miss.reshape(100, -1)
    # train_mask60miss_mat = train_mask60miss.reshape(100, -1)
    # train_speed_data80miss_mat = train_speed_data80miss.reshape(100, -1)
    # train_mask80miss_mat = train_mask80miss.reshape(100, -1)



    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    true_all_speed_feature_mat_scaled = scaler.fit_transform(true_all_speed_feature_mat)

    scaler4weekday = StandardScaler()
    scaler4weekend = StandardScaler()

    # mask20_mat = train_mask20miss_mat ^ true_all_mask_mat
    # mask40_mat = train_mask40miss_mat ^ true_all_mask_mat
    # mask60_mat = train_mask60miss_mat ^ true_all_mask_mat
    # mask80_mat = train_mask80miss_mat ^ true_all_mask_mat
    # base_path = './data/seattle/random_missing'
    # data_path = 'seattle_random_missing20.npy'
    # data_mat = np.load(os.path.join(base_path, data_path))

    # data_ten = np.reshape(data_mat, [365, 288, 323])

    # print("工作日的数据，前面是随机缺失的，后面是非随机缺失的")
    # weekday_all_data = np.load(r'./data/weekday/weekday_all_data.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday_all_data = np.load(r'./data/weekday/mask_weekday_all_data.npy')[:,:,roadsectId_list].reshape([10, -1])
    # weekday_all_data_scaled = scaler4weekday.fit_transform(weekday_all_data)
    #
    # weekday20miss = np.load(r'./data/weekday/weekday20miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # weekday40miss = np.load(r'./data/weekday/weekday40miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # weekday60miss = np.load(r'./data/weekday/weekday60miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # weekday80miss = np.load(r'./data/weekday/weekday80miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday20miss = np.load(r'./data/weekday/mask_weekday20miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday40miss = np.load(r'./data/weekday/mask_weekday40miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday60miss = np.load(r'./data/weekday/mask_weekday60miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday80miss = np.load(r'./data/weekday/mask_weekday80miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    #
    # weekday20nr_miss = np.load(r'./data/weekday/weekday20no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # weekday40nr_miss = np.load(r'./data/weekday/weekday40no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # weekday60nr_miss = np.load(r'./data/weekday/weekday60no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # weekday80nr_miss = np.load(r'./data/weekday/weekday80no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday20nr_miss = np.load(r'./data/weekday/mask_weekday20no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday40nr_miss = np.load(r'./data/weekday/mask_weekday40no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday60nr_miss = np.load(r'./data/weekday/mask_weekday60no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])
    # mask_weekday80nr_miss = np.load(r'./data/weekday/mask_weekday80no_random_miss.npy')[:,:,roadsectId_list].reshape([10, -1])

    print("随机缺失，处理周末数据")
    weekend_all_data = np.load(r'./data/weekend/weekend_all_data.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend_all_data = np.load(r'./data/weekend/mask_weekend_all_data.npy')[:,:,roadsectId_list].reshape([4, -1])
    weekend_all_data_scaled = scaler4weekend.fit_transform(weekend_all_data)
    #
    weekend20miss = np.load(r'./data/weekend/weekend20miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    weekend40miss = np.load(r'./data/weekend/weekend40miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    weekend60miss = np.load(r'./data/weekend/weekend60miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    weekend80miss = np.load(r'./data/weekend/weekend80miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend20miss = np.load(r'./data/weekend/mask_weekend20miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend40miss = np.load(r'./data/weekend/mask_weekend40miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend60miss = np.load(r'./data/weekend/mask_weekend60miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend80miss = np.load(r'./data/weekend/mask_weekend80miss.npy')[:,:,roadsectId_list].reshape([4, -1])

    weekend20nr_miss = np.load(r'./data/weekend/weekend20no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    weekend40nr_miss = np.load(r'./data/weekend/weekend40no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    weekend60nr_miss = np.load(r'./data/weekend/weekend60no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    weekend80nr_miss = np.load(r'./data/weekend/weekend80no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend20nr_miss = np.load(r'./data/weekend/mask_weekend20no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend40nr_miss = np.load(r'./data/weekend/mask_weekend40no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend60nr_miss = np.load(r'./data/weekend/mask_weekend60no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])
    mask_weekend80nr_miss = np.load(r'./data/weekend/mask_weekend80no_random_miss.npy')[:,:,roadsectId_list].reshape([4, -1])

    h_dim = [400]
    ret_str = ''
    for i in h_dim:
        sae = StackedAutoEncoder(2,  [[288*500, 500], [500, 30]])
        sae.model.summary()

        per_train_epochs = 2000
        whole_train_epochs = 1200
        pre_train_lr = 0.00001
        whole_train_lr = 0.000005

        # train_layers(sae, per_train_epochs=per_train_epochs, whole_train_epochs=whole_train_epochs,
        #              pre_train_lr=pre_train_lr, whole_train_lr=whole_train_lr,
        #              pre_train_batch_size=5, whole_train_batch_size=5, train_data=true_all_speed_feature_mat_scaled[:10])
        # random20 = get_mape_rmse(scaler, train_speed_data20miss_mat[10:], sae, true_all_speed_feature_mat[10:],
        #                          train_mask20miss_mat[10:], true_all_mask_mat[10:], 20)
        # random40 = get_mape_rmse(scaler, train_speed_data40miss_mat[10:], sae, true_all_speed_feature_mat[10:],
        #                          train_mask40miss_mat[10:] , true_all_mask_mat[10:], 40)
        # random60 = get_mape_rmse(scaler, train_speed_data60miss_mat[10:], sae, true_all_speed_feature_mat[10:],
        #                          train_mask60miss_mat[10:] , true_all_mask_mat[10:], 60)
        # random80 = get_mape_rmse(scaler, train_speed_data80miss_mat[10:], sae, true_all_speed_feature_mat[10:],
        #                          train_mask80miss_mat[10:] , true_all_mask_mat[10:], 80)
        #
        # no_random20 = get_mape_rmse(scaler, train_speed_data20nr_miss.reshape([14,-1])[10:],
        #                             sae, true_all_speed_feature_mat[10:],
        #                             train_mask20nr_miss.reshape([14,-1])[10:] , true_all_mask_mat[10:],20)
        # no_random40 = get_mape_rmse(scaler, train_speed_data40nr_miss.reshape([14, -1])[10:], sae,
        #                             true_all_speed_feature_mat[10:],
        #                             train_mask40nr_miss.reshape([14, -1])[10:] , true_all_mask_mat[10:], 40)
        # no_random60 = get_mape_rmse(scaler, train_speed_data60nr_miss.reshape([14, -1])[10:], sae,
        #                             true_all_speed_feature_mat[10:],
        #                             train_mask60nr_miss.reshape([14, -1])[10:] , true_all_mask_mat[10:], 60)
        # no_random80 = get_mape_rmse(scaler, train_speed_data80nr_miss.reshape([14, -1])[10:], sae,
        #                             true_all_speed_feature_mat[10:],
        #                             train_mask80nr_miss.reshape([14, -1])[10:] , true_all_mask_mat[10:], 80)

        # print("随机缺失")
        # print(random20)
        # print(random40)
        # print(random60)
        # print(random80)
        # print("非随机缺失")
        # print(no_random20)
        # print(no_random40)
        # print(no_random60)
        # print(no_random80)

        # train_layers(sae, per_train_epochs=per_train_epochs, whole_train_epochs=whole_train_epochs,
        #              pre_train_lr=pre_train_lr, whole_train_lr=whole_train_lr,
        #              pre_train_batch_size=3, whole_train_batch_size=3, train_data=weekday_all_data_scaled[:7])
        # 工作日数据，随机缺失与非随机缺失。
        # res_str20weekday = get_mape_rmse(scaler4weekday, weekday20nr_miss[7:], sae, weekday_all_data[7:],
        #                           mask_weekday20miss[7:] , mask_weekday_all_data[7:],20)
        # res_str40weekday = get_mape_rmse(scaler4weekday, weekday40miss[7:], sae, weekday_all_data[7:],
        #                           mask_weekday40miss[7:] ^ mask_weekday_all_data[7:],40)
        # res_str60weekday = get_mape_rmse(scaler4weekday, weekday60miss[7:], sae, weekday_all_data[7:],
        #                           mask_weekday60miss[7:] ^ mask_weekday_all_data[7:],60)
        # res_str80weekday = get_mape_rmse(scaler4weekday, weekday80miss[7:], sae, weekday_all_data[7:],
        #                           mask_weekday80miss[7:] ^ mask_weekday_all_data[7:],80)
        # print(res_str20weekday)
        # print(res_str40weekday)
        # print(res_str60weekday)
        # print(res_str80weekday)
        # res_str20weekday_nr = get_mape_rmse(scaler4weekday, weekday20nr_miss[7:], sae, weekday_all_data[7:],
        #                                  mask_weekday20nr_miss[7:] , mask_weekday_all_data[7:],20)
        # res_str40weekday_nr = get_mape_rmse(scaler4weekday, weekday40nr_miss[7:], sae, weekday_all_data[7:],
        #                                  mask_weekday40nr_miss[7:] ^ mask_weekday_all_data[7:],40)
        # res_str60weekday_nr = get_mape_rmse(scaler4weekday, weekday60nr_miss[7:], sae, weekday_all_data[7:],
        #                                  mask_weekday60nr_miss[7:] ^ mask_weekday_all_data[7:],60)
        # res_str80weekday_nr = get_mape_rmse(scaler4weekday, weekday80nr_miss[7:], sae, weekday_all_data[7:],
        #                                  mask_weekday80nr_miss[7:] ^ mask_weekday_all_data[7:],80)
        # print(res_str20weekday_nr)
        # print(res_str40weekday_nr)
        # print(res_str60weekday_nr)
        # print(res_str80weekday_nr)
        #
        print("处理周末数据，随机缺失")
        # sae2 = StackedAutoEncoder(2, [[288 * 4248, 256], [256, 128]])
        train_layers(sae, per_train_epochs=per_train_epochs, whole_train_epochs=whole_train_epochs,
                     pre_train_lr=pre_train_lr, whole_train_lr=whole_train_lr,
                     pre_train_batch_size=3, whole_train_batch_size=3, train_data=weekend_all_data_scaled[:3])
        res_str20weekend = get_mape_rmse(scaler4weekend, weekend20miss[3:], sae, weekend_all_data[3:],
                                  mask_weekend20miss[3:] , mask_weekend_all_data[3:],20)
        # res_str40weekend = get_mape_rmse(scaler4weekend, weekend40miss[3:], sae2, weekend_all_data[3:],
        #                           mask_weekend40miss[3:] ^ mask_weekend_all_data[3:],40)
        # res_str60weekend = get_mape_rmse(scaler4weekend, weekend60miss[3:], sae2, weekend_all_data[3:],
        #                           mask_weekend60miss[3:] ^ mask_weekend_all_data[3:],60)
        # res_str80weekend = get_mape_rmse(scaler4weekend, weekend80miss[3:], sae2, weekend_all_data[3:],
        #                           mask_weekend80miss[3:] ^ mask_weekend_all_data[3:],80)
        # print(res_str20weekend)
        # print(res_str40weekend)
        # print(res_str60weekend)
        # print(res_str80weekend)
        #
        # print("周末数据，非随机缺失")
        res_str20weekend_nr = get_mape_rmse(scaler4weekend, weekend20nr_miss[3:], sae, weekend_all_data[3:],
                                         mask_weekend20nr_miss[3:] , mask_weekend_all_data[3:],20)
        # res_str40weekend_nr = get_mape_rmse(scaler4weekend, weekend40nr_miss[3:], sae2, weekend_all_data[3:],
        #                                  mask_weekend40nr_miss[3:] ^ mask_weekend_all_data[3:],40)
        # res_str60weekend_nr = get_mape_rmse(scaler4weekend, weekend60nr_miss[3:], sae2, weekend_all_data[3:],
        #                                  mask_weekend60nr_miss[3:] ^ mask_weekend_all_data[3:],60)
        # res_str80weekend_nr = get_mape_rmse(scaler4weekend, weekend80nr_miss[3:], sae2, weekend_all_data[3:],
        #                                  mask_weekend80nr_miss[3:] ^ mask_weekend_all_data[3:],80)
        # print(res_str20weekend_nr)
        # print(res_str40weekend_nr)
        # print(res_str60weekend_nr)
        # print(res_str80weekend_nr)


        # # 20%
        # train_speed_data20miss_mat = scaler.transform(train_speed_data20miss_mat)
        # pret_data = sae.model.predict(train_speed_data20miss_mat[10:])
        # pret_data = scaler.inverse_transform(pret_data)
        # true_all_speed_feature_mat = scaler.inverse_transform(true_all_speed_feature_mat)
        # true_all_speed_feature_mat = true_all_speed_feature_mat[10:]
        # mask20_mat2 = mask20_mat[10:]
        # a = true_all_speed_feature_mat[mask20_mat2]
        # b = pret_data[mask20_mat2]
        # sae_res2_mape2 = mape(a,b)
        # sae_res2_rmse2 = rmse(a,b)
        # ret_str+="计算缺失位置的补全效果，rank："+str(i)+"\n"
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},mape={}\n".format(
        #      per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res2_mape2)
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},rmse={}\n".format(
        #     per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res2_rmse2)

        # 40%
        # train_speed_data40miss_mat = scaler.transform(train_speed_data40miss_mat)
        # pret_data40 = sae.model.predict(train_speed_data40miss_mat[10:])
        # pret_data40 = scaler.inverse_transform(pret_data40)
        # mask40_mat2 = mask40_mat[10:]
        # a4 = true_all_speed_feature_mat[mask40_mat2]
        # b4 = pret_data40[mask40_mat2]
        # sae_res40_mape2 = mape(a4, b4)
        # sae_res40_rmse2 = rmse(a4, b4)
        # ret_str += "计算缺失位置的补全效果，rank：" + str(i) + "\n"
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},mape={}\n".format(
        #      per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res40_mape2)
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},rmse={}\n".format(
        #      per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res40_rmse2)

        # 60%
        # train_speed_data60miss_mat = scaler.transform(train_speed_data60miss_mat)
        # pret_data60 = sae.model.predict(train_speed_data60miss_mat[10:])
        # pret_data60 = scaler.inverse_transform(pret_data60)
        # mask60_mat2 = mask60_mat[10:]
        # a6 = true_all_speed_feature_mat[mask60_mat2]
        # b6 = pret_data60[mask60_mat2]
        # sae_res60_mape2 = mape(a6, b6)
        # sae_res60_rmse2 = rmse(a6, b6)
        # ret_str += "计算缺失位置的补全效果，rank：" + str(i) + "\n"
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},mape={}\n".format(
        #     per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res60_mape2)
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},rmse={}\n".format(
        #      per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res60_rmse2)

        # 80%
        # train_speed_data80miss_mat = scaler.transform(train_speed_data80miss_mat)
        # pret_data80 = sae.model.predict(train_speed_data80miss_mat[10:])
        # pret_data80 = scaler.inverse_transform(pret_data80)
        # mask80_mat2 = mask80_mat[10:]
        # a8 = true_all_speed_feature_mat[mask80_mat2]
        # b8 = pret_data80[mask80_mat2]
        # sae_res8_mape2 = mape(a8, b8)
        # sae_res8_rmse2 = rmse(a8, b8)
        # ret_str += "计算缺失位置的补全效果，rank：" + str(i) + "\n"
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},mape={}\n".format(
        #     per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res8_mape2)
        # ret_str += "sae_data:,per_train_epochs:{},whole_train_epochs:{},pre_train_lr:{},whole_train_lr:{},rmse={}\n".format(
        #     per_train_epochs, whole_train_epochs, pre_train_lr, whole_train_lr, sae_res8_rmse2)
    # print(ret_str)

# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,mape=26.756664165967592
# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,rmse=2.4338877789576285
# 计算缺失位置的补全效果，rank：400
# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,mape=28.65111325630932
# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,rmse=2.6216131006751877
# 计算缺失位置的补全效果，rank：400
# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,mape=31.007812444613347
# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,rmse=2.860700875803528
# 计算缺失位置的补全效果，rank：400
# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,mape=34.11803039484859
# sae_data:seattle_random_missing20.npy,per_train_epochs:150,whole_train_epochs:100,pre_train_lr:5e-05,whole_train_lr:5e-05,rmse=3.1535257380311656
