import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Bidirectional, Conv1D, Concatenate, Dropout
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from baselines.gat_layer3 import GraphAttentionLayer, TimeGAT

# from graph_attention_layer import GraphAttention
# from utils import get_deviation, get_roadId_list_and_adj_mat

sys.path.append('./baselines')
from baselines.metrics import mape, rmse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]  # 用gpu跑，并且动态分配内存
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 用gpu跑内存不够，用cpu可以跑
# # tf.config.set_soft_device_placement(True)
# # tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental_run_functions_eagerly(True)


# adj = np.load('./sparse_adj.npy', allow_pickle=True)
# 13个维度，分别是 1是速度，第十三维（倒数第一）是属于一天的那个时间段（0~287），第十二维（倒数第二）是周几（0~6），
# 第二维是道路长度特征，第三到六维是方向特征（北-南，南-北，东-西，西-东），第七到十维是道路类型维度，第十一维是拥堵速度特征
def get_model(adj,
              feature_of_edge,
              input_shape,
              f_=5,
              gru_output_dim=12,
              conv1d_filters=12,
              conv1d_kernel_size=5,
              conv1d_strides=1,
              gat_heads=2,
              bigru_merge_mode='concat',
              conv1d_padding='same'):  # decoder改用全连接

    x_in = Input(shape=input_shape[1:])
    # print(x_in.shape)
    # print("adj.shape",adj.shape)
    # speed = x_in[..., 0]
    # speed = tf.keras.layers.Softmax()(speed)
    # road_feature = x_in[..., 1:]

    # mask = speed != 0
    # speed_sum = tf.math.reduce_sum(speed,axis=1,keepdims=True)
    # speed_no_zero_nums = tf.math.count_nonzero(speed,axis=1,dtype=tf.float32,keepdims=True)
    # speed_mean = tf.math.divide(speed_sum,speed_no_zero_nums)
    # tmp_mask = tf.math.is_finite(speed_mean)
    # mean_value = tf.math.reduce_mean(tf.boolean_mask(speed_mean, tmp_mask))
    # speed_mean = tf.where(tmp_mask,speed_mean,mean_value)
    # new_speed_mean = tf.where(mask, speed, speed_mean)
    # new_feature = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(new_speed_mean,axis=-1), road_feature])
    # new_speed_mean_mat = tf.reshape(speed, [-1, 288*500])

    # new_speed_mean_softmax = tf.keras.layers.BatchNormalization()(speed)
    # new_speed_mean_softmax = tf.reshape(new_speed_mean_softmax, [-1, 288, 500])

    # speed_dropout = tf.keras.layers.Dropout(rate=0.35, noise_shape=[1, 500])(speed, training=True)
    # speed_dropout = tf.keras.layers.Dropout(rate=0.35)(speed, training=True)

    # x_in1 = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(speed_dropout, axis=-1), road_feature])
    # x_in1 = tf.keras.layers.Softmax()(x_in1)
    # x_in1 = x_in
    # gat = TimeGAT(x_in.shape[-1], f_, adj, 100)(x_in1)

    # indices = []
    # values = []
    # dense_shape = adj.shape
    # for i in range(len(adj)):
    #     for j in range(len(adj[i])):
    #         if adj[i][j] == 1:
    #             indices.append([i, j])
    #             values.append(adj[i][j])
    # adj_list = [indices, values, dense_shape]
    # print("zhe li you wen ti ma?")
    gat = GraphAttentionLayer(x_in.shape[-1], f_, adj, input_shape[2], feature_of_edge, attn_heads=gat_heads)(x_in)
    # print(" you wen ti ")


    gat_res = tf.transpose(gat, [0, 2, 1, 3])  # gat
    x_in1_t = tf.transpose(x_in, [0, 2, 1, 3])


    forward_layer = GRU(gru_output_dim, return_sequences=True)
    bi_gru = Bidirectional(forward_layer, merge_mode=bigru_merge_mode)
    bi_gru_res = tf.keras.layers.TimeDistributed(bi_gru)(x_in1_t)
    conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                    strides=conv1d_strides, padding=conv1d_padding)(x_in1_t)  # (gat_res)   #(x_in1_t)   #
    conv1d_2 = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                      strides=conv1d_strides, padding=conv1d_padding)(conv1d)  # input_shape=input_shape[2:]

    print(gat_res.shape,bi_gru_res.shape,conv1d_2.shape)

    # concat = Concatenate(axis=-1)([bi_gru_res, conv1d])
    # my model  mape: 33.37859780856408,rmse: 2.530387022823252
    # concat = Concatenate(axis=-1)([tf.expand_dims(x_in1_t[...,0],axis=-1),gat_res, bi_gru_res, conv1d_2])
    # print(gat_res.shape, bi_gru_res.shape, conv1d_2.shape)

    concat = Concatenate(axis=-1)([gat_res,bi_gru_res,conv1d_2])

    dense = Dense(units=512, use_bias=True)(concat)
    dense = tf.keras.layers.LeakyReLU()(dense)
    # dense = tf.keras.layers.Dropout(rate=0.1)(dense)
    dense2 = Dense(units=256, use_bias=True)(dense)
    dense2 = tf.keras.layers.LeakyReLU()(dense2)
    # dense2 = tf.keras.layers.Dropout(rate=0.1)(dense2)
    dense3 = Dense(units=1, use_bias=True)(dense2)


    ret = tf.transpose(dense3, [0, 2, 1, 3])
    # ret = gat

    # ret = tf.transpose(ret, [0, 2, 1, 3])
    model = Model(inputs=x_in, outputs=ret)

    return model


def my_loss(y_true, y_pred):
    mask = y_true != 0
    return tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_true[mask], y_pred[mask])), axis=-1)


def load_data(filename, data_type):
    road_feature=None
    road_feature_filename_type = filename.split(".")[-1]
    if road_feature_filename_type == "npy":
        try:
            road_feature = np.load(filename).astype(data_type)
        except Exception:
            print("文件内容有误，请检查输入数据")
    elif road_feature_filename_type == "csv":
        try:
            df = pd.read_csv(filename, dtype=float, delimiter=' ')
            road_feature = np.array(df).astype(data_type)
        except Exception:
            print("文件内容有误，请检查输入数据")
    elif road_feature_filename_type in ["xls", "xlsx"]:
        try:
            df = pd.read_excel(filename, header=None, dtype=float)
            road_feature = np.array(df).astype(data_type)
        except Exception:
            print("文件内容有误，请检查输入数据")
    elif road_feature_filename_type == "txt":
        try:
            road_feature = np.loadtxt(filename, delimiter=' ', dtype=np.float).astype(data_type)
        except Exception:
            print("文件内容有误，请检查输入数据")
    else:
        print("暂不支持非[npy、csv、txt、xls、xlsx]文件")

    return road_feature
# if __name__ == "__main__":
def GSTCompletor(train_data, road_feature_filename: str, feature_of_edge_filename, roadsect_adj_mat_filename, time_slot, learning_rate=0.005,
                 n_attn_heads=1,
                 epochs=700,
                 batch_size=2):

    # 要求train_data为二维矩阵，行为路段，列为时间
    # road_feature为二维矩阵，的行为路段数，列为路段的特征数
    # feature_of_edge为二维矩阵，行为路段数，列为边的特征数
    # roadssect_adj_mat为二维方阵，方阵中的值为0或1，i行j列值为1表示道路i和道路j有直接连接，否则为无连接。
    # time_slot为大于1的正整数的值，表示训练数据中，时间可以按time_slot分成一个一个时间段。

    road_feature = load_data(road_feature_filename,"int")
    feature_of_edge = load_data(feature_of_edge_filename,"int")
    roadsect_adj_mat = load_data(roadsect_adj_mat_filename,"int")
    print("road_feature shape",road_feature.shape,"feature_of_edge shape",feature_of_edge.shape,"roadsect_adj_mat shape",roadsect_adj_mat.shape)
    print("train_data shape",train_data.shape)
    road_num = train_data.shape[0]
    if train_data.shape[-1] % time_slot != 0:
        print("参数错误，time_slot参数有误")
        return None
    elif road_num != road_feature.shape[0]:
        print("参数错误，road_feature数据有误")
        return None
    elif roadsect_adj_mat.ndim!=2 or roadsect_adj_mat.shape[0]!=roadsect_adj_mat.shape[1] or roadsect_adj_mat.shape[0]!=road_num:
        print("参数错误，临界矩阵文件数据有误")
        return None
    elif feature_of_edge.ndim!=2 or feature_of_edge.shape[0]!=feature_of_edge.shape[1] or feature_of_edge.shape[0]!= road_num:
        print("参数错误，边特征文件数据有误")
        return None
    day_slot = train_data.shape[-1] // time_slot
    road_feature_num = road_feature.shape[-1]
    model = get_model(roadsect_adj_mat, feature_of_edge,input_shape=[day_slot,time_slot,road_num,road_feature_num+1])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=my_loss,  # 'mean_squared_error',  # 'mean_squared_error', # 如果是输入多天的数据，loss就不好用自定义的了
                  weighted_metrics=['acc'],
                  )
    # print("输出模型")
    model.summary()
    tmp_nparr = np.tile(road_feature, [day_slot * time_slot, 1]).reshape(
        [day_slot, time_slot, road_num, road_feature_num])
    for _ in range(epochs):
        train_feature = np.where(np.random.random(train_data.shape) < 0.2, 0., train_data)

        train_feature = np.reshape(train_feature, [road_num, day_slot, time_slot]).transpose([1, 2, 0])
        train_feature = np.concatenate([np.expand_dims(train_feature, -1), tmp_nparr], -1)
        model.fit(train_feature,
                  # [0,2,3,4,5,6,7,8,9]],#[..., [0,2,3,4,5]],  # all_feature100[:10], # true_all_speed_data100[:10],   #
                  np.expand_dims(np.transpose(np.reshape(train_data, [road_num, day_slot, time_slot]), [1, 2, 0]), -1),
                  # true_all_speed_data_scaled[:10],
                  epochs=1,  # epochs 300
                  batch_size=batch_size,
                  # validation_data=validation_data,
                  shuffle=True,  # Shuffling data means shuffling the whole graph
                  # callbacks=[mc_callback, tb_callback],  # callbacks=[es_callback, tb_callback, mc_callback]
                  )
    model_pret = model.predict(np.concatenate(
        [np.expand_dims(np.reshape(train_data, [road_num, day_slot, time_slot]).transpose([1, 2, 0]), -1),tmp_nparr], -1))
    res = np.where(train_data!=0,train_data,np.transpose(np.squeeze(model_pret).reshape([time_slot*day_slot,road_num]),[1,0]))
    return res

