import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Bidirectional, Conv1D, Concatenate, Dropout
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gat_layer3 import GraphAttentionLayer, TimeGAT
from graph_attention_layer import GraphAttention
from utils import get_deviation, get_roadId_list_and_adj_mat

sys.path.append('./baselines')
from baselines.metrics import mape, rmse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]  # 用gpu跑，并且动态分配内存
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 用gpu跑内存不够，用cpu可以跑
# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental_run_functions_eagerly(True)


# adj = np.load('./sparse_adj.npy', allow_pickle=True)
# 13个维度，分别是 1是速度，第十三维（倒数第一）是属于一天的那个时间段（0~287），第十二维（倒数第二）是周几（0~6），
# 第二维是道路长度特征，第三到六维是方向特征（北-南，南-北，东-西，西-东），第七到十维是道路类型维度，第十一维是拥堵速度特征
def get_model(adj,
              feature_of_edge,
              f_=5,
              gru_output_dim=12,
              conv1d_filters=12,
              conv1d_kernel_size=5,
              conv1d_strides=1,
              gat_heads=2,
              bigru_merge_mode='concat',
              conv1d_padding='same'):  # decoder改用全连接

    x_in = Input(shape=(288, 500, 11))
    speed = x_in[...,0]
    # speed = tf.keras.layers.Softmax()(speed)
    road_feature = x_in[...,1:]

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

    speed_dropout = tf.keras.layers.Dropout(rate=0.35,noise_shape=[1,500])(speed, training=True)
    # speed_dropout = tf.keras.layers.Dropout(rate=0.35)(speed, training=True)

    x_in1 = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(speed_dropout,axis=-1),road_feature])
    # x_in1 = tf.keras.layers.Softmax()(x_in1)
    # x_in1 = x_in
    # gat = TimeGAT(x_in.shape[-1], f_, adj, 100)(x_in1)

    gat = GraphAttentionLayer(x_in.shape[-1], f_, adj,  500, feature_of_edge, attn_heads=gat_heads)(x_in1)

    # gat = GraphAttentionLayer(gat.shape[-1], f_, adj,  500, feature_of_edge, attn_heads=gat_heads)(gat)

    # GraphAttentionLayer output shape (None, 288,4248,f_)
    # transpose 为 (None,4248,288,f_)
    # x_in2 = tf.keras.layers.Dropout(rate=0.2)(x_in, training=True)
    # gat2 = tf.keras.layers.Dropout(rate=0.2)(gat, training=True)
    # x_in3 = tf.keras.layers.BatchNormalization(trainable=False)(x_in2)

    gat_res = tf.transpose(gat, [0, 2, 1, 3])   # gat
    x_in1_t = tf.transpose(x_in1, [0, 2, 1, 3])

    # new_feature_t = tf.transpose(new_feature, [0,2,1,3])
    # gat_res = tf.keras.layers.Dropout(rate=0.3)(x_in2, training=True)

    # gat_res = tf.keras.layers.Softmax()(gat_res)
    # gat_res2 = tf.reshape(gat_res, [-1, 288, 1])

    forward_layer = GRU(gru_output_dim, return_sequences=True)
    bi_gru = Bidirectional(forward_layer, merge_mode=bigru_merge_mode)
    bi_gru_res = tf.keras.layers.TimeDistributed(bi_gru)(x_in1_t)   # (gat_res)   #(x_in1_t)
    # bi_gru = tf.reshape(bi_gru, [-1, 4248, 288, gru_output_dim])
    conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                    strides=conv1d_strides, padding=conv1d_padding)(x_in1_t)   # (gat_res)   #(x_in1_t)   #
    conv1d_2 = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                      strides=conv1d_strides, padding=conv1d_padding)(conv1d)  # input_shape=input_shape[2:]

    # concat = Concatenate(axis=-1)([bi_gru_res, conv1d])
    # my model  mape: 33.37859780856408,rmse: 2.530387022823252
    # concat = Concatenate(axis=-1)([tf.expand_dims(x_in1_t[...,0],axis=-1),gat_res, bi_gru_res, conv1d_2])
    # print(gat_res.shape, bi_gru_res.shape, conv1d_2.shape)

    concat = Concatenate(axis=-1)([gat_res, bi_gru_res, conv1d_2])  # ([bi_gru_res, conv1d_2])  #
    # concat = Concatenate(axis=-1)([bi_gru_res, conv1d_2])  # ([bi_gru_res, conv1d_2])  #


    # print(concat.shape,[-1]+concat.shape.as_list()[1:-2]+[(18+36+18)*288])
    # concat = Concatenate(axis=-1)([tf.expand_dims(tf.transpose(x_in[...,0], [0,2,1]), axis=-1),bi_gru_res, conv1d])
    # concat = tf.reshape(concat, [-1]+concat.shape.as_list()[1:-2]+[(18+36+18)*288])
    dense = Dense(units=512, use_bias=True)(concat)
    dense = tf.keras.layers.LeakyReLU()(dense)
    # dense = tf.keras.layers.Dropout(rate=0.1)(dense)
    dense2 = Dense(units=256, use_bias=True)(dense)
    dense2 = tf.keras.layers.LeakyReLU()(dense2)
    # dense2 = tf.keras.layers.Dropout(rate=0.1)(dense2)
    dense3 = Dense(units=1, use_bias=True)(dense2)

    # ret = tf.transpose(tf.reshape(dense3, [-1, 4248, 288, 1]), [0, 2, 1, 3])
    ret = tf.transpose(dense3, [0, 2, 1, 3])

    # ret = tf.transpose(ret, [0, 2, 1, 3])
    model = Model(inputs=x_in, outputs=ret)

    return model

def my_loss(y_true, y_pred):
    mask = y_true != 0
    return tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_true[mask], y_pred[mask])), axis=-1)

def get_rmse_mape(mod, a, b, m1, m2):

    pret = mod.predict(a)
    pret = np.squeeze(pret,axis=-1)
    b = np.squeeze(b,axis=-1)
    # print(a.shape,b.shape,pret.shape,m1.shape,m2.shape)
    m1 = np.squeeze(m1,axis=-1)
    m = m1 ^ m2

    print(np.sum(m))
    mape_res20 = mape(b[m], pret[m])
    rmse_res20 = rmse(b[m], pret[m])
    return_str = 'my model  mape: {},rmse: {}'.format(mape_res20, rmse_res20)
    print(return_str)

    deviation_true_data, deviation_pret_data = get_deviation(b, m1,
                                                             pret)
    print("BTRMF_deviation_pret_data", deviation_pret_data)
    print("deviation_true_data", deviation_true_data)

    sorted_list = np.load(r'./data/sorted_var_500_road.npy')
    cao = b[...,sorted_list[400:]]
    nima = pret[...,sorted_list[400:]]
    mlgb = m[...,sorted_list[400:]]
    BTRMF_res2_mape2_big_var = mape(cao[mlgb], nima[mlgb])
    BTRMF_res2_rmse2_big_var = rmse(cao[mlgb], nima[mlgb])
    cao = b[..., sorted_list[:400]]
    nima = pret[..., sorted_list[:400]]
    mlgb = m[..., sorted_list[:400]]
    BTRMF_res2_mape2_small_var = mape(cao[mlgb], nima[mlgb])
    BTRMF_res2_rmse2_small_var = rmse(cao[mlgb], nima[mlgb])

    print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)


    b_ = np.reshape(b[m],[-1])
    pret_ = np.reshape(pret[m],[-1])
    _, id_list = zip(*sorted(zip(np.abs(np.subtract(b_, pret_)),range(len(b_))),key=lambda x:x[0]))
    id_list = list(id_list)
    true_data_np_array = b_[id_list]
    pret_data_np_array = pret_[id_list]
    mape_res202 = mape(true_data_np_array[:int(len(id_list)*0.9)], pret_data_np_array[:int(len(id_list)*0.9)])
    rmse_res202 = rmse(true_data_np_array[:int(len(id_list)*0.9)], pret_data_np_array[:int(len(id_list)*0.9)])
    mape_res20_95 = mape(true_data_np_array[:int(len(id_list) * 0.95)], pret_data_np_array[:int(len(id_list) * 0.95)])
    rmse_res20_95 = rmse(true_data_np_array[:int(len(id_list) * 0.95)], pret_data_np_array[:int(len(id_list) * 0.95)])
    mape_res20_97 = mape(true_data_np_array[:int(len(id_list) * 0.97)], pret_data_np_array[:int(len(id_list) * 0.97)])
    rmse_res20_97 = rmse(true_data_np_array[:int(len(id_list) * 0.97)], pret_data_np_array[:int(len(id_list) * 0.97)])
    mape_res20_99 = mape(true_data_np_array[:int(len(id_list) * 0.85)], pret_data_np_array[:int(len(id_list) * 0.85)])
    rmse_res20_99 = rmse(true_data_np_array[:int(len(id_list) * 0.85)], pret_data_np_array[:int(len(id_list) * 0.85)])
    # mape_res202 = mape(b_, pret_)
    # rmse_res202 = rmse(b_, pret_)
    return_str2 = 'my model  mape_90: {},rmse_90: {}'.format(mape_res202, rmse_res202)
    return_str2 += 'my model  mape_95: {},rmse_95: {}'.format(mape_res20_95, rmse_res20_95)
    return_str2 += 'my model  mape_97: {},rmse_97: {}'.format(mape_res20_97, rmse_res20_97)
    return_str2 += 'my model  mape_99: {},rmse_99: {}'.format(mape_res20_99, rmse_res20_99)
    print(return_str2)

    speed_feature_ten = b
    data_ten = pret
    mask_ten = m
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
    print(str2)

if __name__ == "__main__":
    # 选出的100条道路
    selected100road_list_and_adj_mat = np.load("./data/selected500road_index_and_adj_mat.npz")
    roadsectId_list = selected100road_list_and_adj_mat['arr_0']     # 被选中的道路在原先4248条道路中的编号
    roadsect_adj_mat = selected100road_list_and_adj_mat['arr_1']    # 被选中的道路的邻接矩阵

    feature_of_edge = np.load("./data/feature_of_edge.npy")

    learning_rate = 0.005  # Learning rate for Adam
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 49.795909892694354,rmse: 5.5290348272584255

    # N = true_feature.shape[1]
    es_patience = 100

    n_attn_heads = 1
    dropout_rate = 0.4
    l2_reg = 5e-4 / 2
    model = get_model(roadsect_adj_mat, feature_of_edge[roadsectId_list,roadsectId_list])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=my_loss,  # 'mean_squared_error',  # 'mean_squared_error', # 如果是输入多天的数据，loss就不好用自定义的了
                  weighted_metrics=['acc'],
                  )
    model.summary()

    # tb_callback = TensorBoard(batch_size=1,log_dir='./logs')

    all_feature = np.load(r'./data/all_feature.npy')  # (14,288,4248,13)
    all_feature100 = all_feature[:, :, roadsectId_list, :]  # (14,288,100,13)

    # scaler = StandardScaler()
    # all_feature_scaled = tf.reshape(scaler.fit_transform(tf.reshape(all_feature100,[-1, 13])),[14,288,500,13])
    true_all_data = np.load('./all_speed_data.npz')  # (14,288,4248,1)
    # shape(14,288,4248,1)
    true_all_speed_data, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    true_all_speed_data100 = true_all_speed_data[:, :, roadsectId_list, :]  # shape(14,288,100,1)
    true_all_mask100 = true_all_mask[:, :, roadsectId_list, :]  # shape(14,288,100,1)
    # true_all_speed_data_scaled = tf.reshape(scaler.fit_transform(true_all_speed_data.reshape([-1,1])), [14,288,4248,1])
    epochs = 7# 700

    # 去掉道路类型 [0,1,2,3,4,5,10]   7
    # 去掉方向 [0,1,6,7,8,9,10]    7
    # 去掉限速 [0,1,2,3,4,5,6,7,8,9]   10
    # 去掉道路长度 [0,2,3,4,5,6,7,8,9,10]     10

    # 完整工作日数据      [0,3,4,5,6,7,10,11,12,13]
    weekday_all_data = np.load(r'./data/weekday/weekday_all_data.npy')[...,roadsectId_list]
    mask_weekday_all_data = np.load(r'./data/weekday/mask_weekday_all_data.npy' )[...,roadsectId_list]

    # 完整周末数据          [1,2,8,9]
    weekend_all_data = np.load(r'./data/weekend/weekend_all_data.npy')[...,roadsectId_list]
    mask_weekend_all_data = np.load(r'./data/weekend/mask_weekend_all_data.npy')[...,roadsectId_list]

    # 随机缺失20%
    # 随机缺失20%，工作日数据
    weekday20miss_rm=np.load(r'./data/weekday/weekday20miss.npy')[...,roadsectId_list]       # shape(10,288,500)
    mask_weekday20miss_rm=np.load(r'./data/weekday/mask_weekday20miss.npy')[...,roadsectId_list]
    train_feature_weekday20miss_rm = np.concatenate([np.expand_dims(weekday20miss_rm,axis=-1),
                                                     all_feature100[[0,3,4,5,6,7,10,11,12,13],:,:,1:-2]],axis=-1)
    # 随机缺失20%，周末数据
    weekend20miss_rm=np.load(r'./data/weekend/weekend20miss.npy')[...,roadsectId_list]
    mask_weekend20miss_rm=np.load(r'./data/weekend/mask_weekend20miss.npy')[...,roadsectId_list]
    train_feature_weekend20miss_rm = np.concatenate([np.expand_dims(weekend20miss_rm, axis=-1),
                                                     all_feature100[[1,2,8,9], :, :, 1:-2]], axis=-1)

    # 非随机缺失20%
    # 非随机缺失20%，工作日数据
    weekday20miss_nm=np.load(r'./data/weekday/weekday20no_random_miss.npy')[...,roadsectId_list]
    mask_weekday20miss_nm=np.load(r'./data/weekday/mask_weekday20no_random_miss.npy')[...,roadsectId_list]
    train_feature_weekday20miss_nm = np.concatenate([np.expand_dims(weekday20miss_nm, axis=-1),
                                                     all_feature100[[0,3,4,5,6,7,10,11,12,13], :, :, 1:-2]], axis=-1)
    # 非随机缺失20%，周末数据
    weekend20miss_nm=np.load(r'./data/weekend/weekend20no_random_miss.npy')[...,roadsectId_list]
    mask_weekend20miss_nm=np.load(r'./data/weekend/mask_weekend20no_random_miss.npy')[...,roadsectId_list]
    train_feature_weekend20miss_nm = np.concatenate([np.expand_dims(weekend20miss_nm, axis=-1),
                                                     all_feature100[[1,2,8,9], :, :, 1:-2]], axis=-1)
    # 先算工作日的
    # 再算周末的
    model.fit(all_feature100[[1,2,8],:,:,:-2],#[0,2,3,4,5,6,7,8,9]],#[..., [0,2,3,4,5]],  # all_feature100[:10], # true_all_speed_data100[:10],   #
              true_all_speed_data100[[1,2,8]],
              # true_all_speed_data_scaled[:10],
              epochs=epochs,  # epochs 300
              batch_size=5,
              # validation_data=validation_data,
              shuffle=True,  # Shuffling data means shuffling the whole graph
              # callbacks=[mc_callback, tb_callback],  # callbacks=[es_callback, tb_callback, mc_callback]
              )
    # get_rmse_mape(model,train_feature_weekday20miss_rm[7:], true_all_speed_data100[[11,12,13]],true_all_mask100[[11,12,13]], mask_weekday20miss_rm[7:])
    # get_rmse_mape(model,train_feature_weekday20miss_nm[7:], true_all_speed_data100[[11,12,13]],true_all_mask100[[11,12,13]], mask_weekday20miss_nm[7:])
    # get_rmse_mape(model, train_feature_weekend20miss_rm[3:], true_all_speed_data100[[9]],
    #               true_all_mask100[[9]], mask_weekend20miss_rm[3:])
    get_rmse_mape(model, train_feature_weekend20miss_nm[3:], true_all_speed_data100[[9]],
                  true_all_mask100[[9]], mask_weekend20miss_nm[3:])

    # # feature_name20rm = "train_feature20random_missing.npy"
    # feature_name20rm = "train_feature20no_random_missing_road.npy"
    # train_feature20rm = np.load('./data/{}'.format(feature_name20rm))   # shape （14,288,4248,13)
    # train_feature100road20rm = train_feature20rm[:, :, roadsectId_list, :]  # shape (14,288,100,13)
    # # feature_name40rm = "train_feature40random_missing.npy"
    # feature_name40rm = "train_feature40no_random_missing_road.npy"
    # train_feature40rm = np.load('./data/{}'.format(feature_name40rm))  # shape （14,288,4248,13)
    # train_feature100road40rm = train_feature40rm[:, :, roadsectId_list, :]  # shape (14,288,100,13)
    # # feature_name60rm = "train_feature60random_missing.npy"
    # feature_name60rm = "train_feature60no_random_missing_road.npy"
    # train_feature60rm = np.load('./data/{}'.format(feature_name60rm))  # shape （14,288,4248,13)
    # train_feature100road60rm = train_feature60rm[:, :, roadsectId_list, :]  # shape (14,288,100,13)
    # # feature_name80rm = "train_feature80random_missing.npy"
    # feature_name80rm = "train_feature80no_random_missing_road.npy"
    # train_feature80rm = np.load('./data/{}'.format(feature_name80rm))  # shape （14,288,4248,13)
    # train_feature100road80rm = train_feature80rm[:, :, roadsectId_list, :]  # shape (14,288,100,13)

    # true_all_speed_data = np.squeeze(true_all_speed_data)   # shape (14,288,4248)
    # true_all_mask = np.squeeze(true_all_mask)               # shape (14,288,4248)

    # # data_name20rm = "all_train_data20random_missing.npz"
    # data_name20rm = "all_train_data20no_random_missing_road.npz"
    # train_data20rm = np.load("./data/{}".format(data_name20rm))
    # train_speed_data20rm = train_data20rm['arr_0']  # shape(14,288,4248)
    # train_mask20rm = train_data20rm['arr_1']  # train_mask shape (14,288,4248)
    # train_speed_data100_20rm = train_speed_data20rm[:, :, roadsectId_list]  # shape (14,288,100)
    # train_mask100_20rm = train_mask20rm[:, :, roadsectId_list]  # train_mask shape (14,288,100)
    #
    # # data_name40rm = "all_train_data40random_missing.npz"
    # data_name40rm = "all_train_data40no_random_missing_road.npz"
    # train_data40rm = np.load("./data/{}".format(data_name40rm))
    # train_speed_data40rm = train_data40rm['arr_0']  # shape(14,288,4248)
    # train_mask40rm = train_data40rm['arr_1']  # train_mask shape (14,288,4248)
    # train_speed_data100_40rm = train_speed_data40rm[:, :, roadsectId_list]  # shape (14,288,100)
    # train_mask100_40rm = train_mask40rm[:, :, roadsectId_list]  # train_mask shape (14,288,100)
    #
    # # data_name60rm = "all_train_data60random_missing.npz"
    # data_name60rm = "all_train_data60no_random_missing_road.npz"
    # train_data60rm = np.load("./data/{}".format(data_name60rm))
    # train_speed_data60rm = train_data60rm['arr_0']  # shape(14,288,4248)
    # train_mask60rm = train_data60rm['arr_1']  # train_mask shape (14,288,4248)
    # train_speed_data100_60rm = train_speed_data60rm[:, :, roadsectId_list]  # shape (14,288,100)
    # train_mask100_60rm = train_mask60rm[:, :, roadsectId_list]  # train_mask shape (14,288,100)
    #
    # # data_name80rm = "all_train_data80random_missing.npz"
    # data_name80rm = "all_train_data80no_random_missing_road.npz"
    # train_data80rm = np.load("./data/{}".format(data_name80rm))
    # train_speed_data80rm = train_data80rm['arr_0']  # shape(14,288,4248)
    # train_mask80rm = train_data80rm['arr_1']  # train_mask shape (14,288,4248)
    # train_speed_data100_80rm = train_speed_data80rm[:, :, roadsectId_list]  # shape (14,288,100)
    # train_mask100_80rm = train_mask80rm[:, :, roadsectId_list]  # train_mask shape (14,288,100)

    # get_rmse_mape(model,train_feature100road20rm[10:,:,:,:-2], true_all_speed_data100[10:],true_all_mask100[10:], train_mask100_20rm[10:])
    # get_rmse_mape(model,train_feature100road40rm[10:,:,:,:-2], true_all_speed_data100[10:],true_all_mask100[10:], train_mask100_40rm[10:])
    # get_rmse_mape(model,train_feature100road60rm[10:,:,:,:-2], true_all_speed_data100[10:],true_all_mask100[10:], train_mask100_60rm[10:])
    # get_rmse_mape(model,train_feature100road80rm[10:,:,:,:-2], true_all_speed_data100[10:],true_all_mask100[10:], train_mask100_80rm[10:])
    # get_rmse_mape(model, train_feature100road20rm[10:, :, :, [0, 2, 3, 4, 5, 6, 7, 8, 9]], true_all_speed_data100[10:],
    #               true_all_mask100[10:], train_mask100_20rm[10:])
    # get_rmse_mape(model, train_feature100road40rm[10:, :, :, [0, 2, 3, 4, 5, 6, 7, 8, 9]], true_all_speed_data100[10:],
    #               true_all_mask100[10:], train_mask100_40rm[10:])
    # get_rmse_mape(model, train_feature100road60rm[10:, :, :, [0, 2, 3, 4, 5, 6, 7, 8, 9]], true_all_speed_data100[10:],
    #               true_all_mask100[10:], train_mask100_60rm[10:])
    # get_rmse_mape(model, train_feature100road80rm[10:, :, :, [0, 2, 3, 4, 5, 6, 7, 8, 9]], true_all_speed_data100[10:],
    #               true_all_mask100[10:], train_mask100_80rm[10:])





