import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Bidirectional, Conv1D, Concatenate, Dropout
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from gat_layer2 import Timeserial_GAT
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

# shape为（288，4248，11）,(288,4248,1)
# true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = get_feature(0.2)
# true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = np.load(
#     './feature.npz').values()
# 在数据本身就缺失的情况下，再丢掉一部分，丢掉的比例是多少？  应该是再丢掉20%

adj = np.load('./sparse_adj.npy', allow_pickle=True)


# _, adj_mat = get_roadId_list_and_adj_mat()
#
# # 把邻接矩阵变成spaesetensor
# indices = []
# values = []
# dense_shape = adj_mat.shape
# for i in range(len(adj_mat)):
#     for j in range(len(adj_mat[i])):
#         if adj_mat[i][j] == 1:
#             indices.append([i, j])
#             values.append(adj_mat[i][j])
# adj = [indices, values, dense_shape]

# adj = adj_mat

class Conv_Bigru(Layer):
    def __init__(self, gru_output_dim=12, bigru_merge_mode='concat', conv1d_filters=12, conv1d_kernel_size=7,
                 conv1d_strides=1, conv1d_padding='same', **kwargs):
        super(Conv_Bigru, self).__init__()
        self.gru_output_dim = gru_output_dim
        self.conv1d_filters = conv1d_filters
        self.conv1d_kernel_size = conv1d_kernel_size
        self.conv1d_strides = conv1d_strides
        self.conv1d_padding = conv1d_padding
        self.bigru_merge_mode = bigru_merge_mode

    def build(self, input_shape):
        self.forward_layer = GRU(self.gru_output_dim, return_sequences=True)
        self.backward_layer = GRU(self.gru_output_dim, return_sequences=True, go_backwards=True)
        self.bigru = Bidirectional(self.forward_layer, backward_layer=self.backward_layer,
                                   merge_mode=self.bigru_merge_mode)

        self.conv1d = Conv1D(filters=self.conv1d_filters, kernel_size=self.conv1d_kernel_size,
                             strides=self.conv1d_strides, padding=self.conv1d_padding)  # input_shape=input_shape[2:]
        self.conv1d_2 = Conv1D(filters=self.conv1d_filters, kernel_size=self.conv1d_kernel_size,
                               strides=self.conv1d_strides, padding=self.conv1d_padding)  # input_shape=input_shape[2:]
        self.concat = Concatenate(axis=-1)
        self.built = True

    def call(self, inputs, training=True):
        bigru_output = self.bigru(inputs)
        conv1d_output = self.conv1d(inputs)
        conv1d_2_output = self.conv1d_2(conv1d_output)
        output = self.concat([bigru_output, conv1d_2_output])
        return output

    def compute_output_shape(self, input_shape):
        pass


def get_model():
    F_ = 9
    gru_output_dim = 12
    conv1d_filters = 12
    conv1d_kernel_size = 7
    conv1d_strides = 1
    conv1d_padding = 'same'
    bigru_merge_mode = 'concat'

    X_in = Input(shape=(288, 4248, 13), batch_size=1)  # batch_size指定时才可以在后面的程序中对第一维进行遍历

    gat = GraphAttention(F_=F_, adj=adj, attn_heads=2)
    # print(X_in.shape[-2:])
    # gat_timedistributed = TimeDistributed(gat)(X_in)        # , input_shape=X_in.shape[-2:]
    gat_ret_list = []
    for i in range(X_in.shape[0]):
        tmp = []
        for j in range(X_in.shape[1]):
            tmp.append(gat(X_in[i, j]))
        t = tf.stack(tmp, axis=0)
        gat_ret_list.append(t)
    gat_ret = tf.stack(gat_ret_list)
    tgat_output = tf.transpose(gat_ret, [0, 2, 1, 3])

    # conv_bigru_timedistributed = TimeDistributed(Conv_Bigru(gru_output_dim=gru_output_dim,
    #                                                         conv1d_filters=conv1d_filters,
    #                                                         conv1d_kernel_size=conv1d_kernel_size,
    #                                                         conv1d_strides=conv1d_strides,
    #                                                         conv1d_padding=conv1d_padding),
    #                                              input_shape=tgat_output.shape[-2:])(
    #     tgat_output)

    conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                    strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
    conv1d_2 = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                      strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
    concat = Concatenate(axis=2)  # ([x, y]), axis=2表示将第三维合并
    dense = Dense(units=32, activation='relu')
    dense2 = Dense(units=16, activation='relu')
    dense3 = Dense(units=1)

    # dense_timedistributed = TimeDistributed(dense, input_shape=conv_bigru_timedistributed.shape[-2:])(conv_bigru_timedistributed)
    # dense2_timedistributed = TimeDistributed(dense2, input_shape=dense_timedistributed.shape[-2:])(dense_timedistributed)
    # dense3_timedistributed = TimeDistributed(dense3, input_shape=dense2_timedistributed.shape[-2:])(dense2_timedistributed)
    # ot = tf.transpose(dense3_timedistributed, [0, 2, 1, 3])

    # decoder
    # forward_layer2 = GRU(32, return_sequences=True)  # 这里好像没法指定输出维度和状态维度不相同
    # backward_layer2 = GRU(32, return_sequences=True, go_backwards=True)
    # decoder_bigru = Bidirectional(forward_layer2, backward_layer=backward_layer2, merge_mode='concat')

    # 对于简单、无状态的自定义操作，你也许可以通过 layers.core.Lambda 层来实现。即使用keras.core.lambda()
    # tgat = Timeserial_GAT(F_, adj)

    # 数据标准化处理
    # speed_data = X_in[:, :, :, 1]
    # # m = tf.concat([tf.ones(speed_data.shape[:-1]+[1]),tf.zeros(speed_data.shape[:-1]+(speed_data.shape[-1]-1))],axis=-1)
    # mean = tf.math.reduce_mean(speed_data)
    # std = tf.math.reduce_std(speed_data)
    # speed_data = tf.math.divide(tf.math.subtract(speed_data, mean), std)
    # # X_in = tf.where(m,tf.tile(speed_data,[1,1,1,X_in.shape[-1]-1]),X_in)
    # X_in_1 = tf.concat([tf.expand_dims(speed_data, axis=-1), X_in[..., 1:]], axis=-1)

    forward_layer = GRU(gru_output_dim, return_sequences=True)
    backward_layer = GRU(gru_output_dim, return_sequences=True, go_backwards=True)
    bigru = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode=bigru_merge_mode)
    output = []
    for i in range(tgat_output.shape[0]):
        # for j in range(tgat_output.shape[1])
        bg = bigru(tgat_output[i])
        con = conv1d(tgat_output[i])
        con2 = conv1d_2(con)
        cc = concat([bg, con2])
        den_out = dense(cc)
        den_out2 = dense2(den_out)
        den_out3 = dense3(den_out2)
        output.append(den_out3)
    ret = tf.stack(output, 0)  # shape of ret is (288,4248,1)
    ot = tf.transpose(ret, [0, 2, 1, 3])

    # 数据反标准化
    # ot = tf.math.add(tf.math.multiply(ret, std), mean)

    model = Model(inputs=X_in, outputs=ot)
    return model

def my_loss(y_true, y_pred):
    mask = y_true != 0
    return tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_true[mask],y_pred[mask])),axis=-1)

def get_model2(F_=7,
               gru_output_dim=12,
               conv1d_filters=12,
               conv1d_kernel_size=7,
               conv1d_strides=1,
               optimizer_lr=0.01,
               gat_heads=2,
               bigru_merge_mode='concat',
               conv1d_padding='same'):  # decoder改用全连接

    X_in = Input(shape=(288, 4248, 13), batch_size=2)  # batch_size指定时才可以在后面的程序中对第一维进行遍历
    forward_layer = GRU(gru_output_dim, return_sequences=True)
    backward_layer = GRU(gru_output_dim, return_sequences=True, go_backwards=True)
    # bigru = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode=bigru_merge_mode)
    # 试试双向gru用同一个GRU。
    bigru = Bidirectional(forward_layer, merge_mode=bigru_merge_mode)
    conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                    strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
    conv1d_2 = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                      strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
    concat = Concatenate(axis=-1)  # ([x, y]), axis=2表示将第三维合并
    dense = Dense(units=32,activation='relu')
    dense2 = Dense(units=16,activation='relu')
    dense3 = Dense(units=1)
    # 对于简单、无状态的自定义操作，你也许可以通过 layers.core.Lambda 层来实现。即使用keras.core.lambda()
    tgat = Timeserial_GAT(F_, adj, gat_heads)

    speed_data = X_in[..., 0]  # shape  （14，288，4248）

    mask = speed_data != 0  # 布尔矩阵，空缺值位置为0，非缺失值位置为1
    # 将每个路段每天的速度作平均填入空缺位置。
    speed_sum = tf.math.reduce_sum(speed_data,axis=1,keepdims=True)
    speed_no_zero_nums = tf.math.count_nonzero(speed_data,axis=1,dtype=tf.float32,keepdims=True)
    speed_mean = tf.math.divide(speed_sum,speed_no_zero_nums)
    tmp_mask = tf.math.is_finite(speed_mean)
    mean_value = tf.math.reduce_mean(tf.boolean_mask(speed_mean, tmp_mask))
    speed_mean = tf.where(tmp_mask,speed_mean,mean_value)
    new_speed_mean = tf.where(mask, speed_data, speed_mean)

    # week_feature = tf.cast(X_in[..., -2], tf.int32)        # shape (14,288,4248)
    # time_slice_feature = tf.cast(X_in[..., -1], tf.int32)
    # week_feature_one_hot = tf.one_hot(week_feature, 7)          # shape(14,288,4248,7)
    # time_slice_feature_one_hot = tf.one_hot(time_slice_feature, 288)        # shape(14,288,4248,288)
    # time_feature = tf.concat([week_feature_one_hot,time_slice_feature_one_hot], axis=-1)
    # # 换轴
    # time_feature = tf.transpose(time_feature,[0,2,1,3])
    # dense4time_feature = Dense(units=4)


    # 沿着最后一个轴算平均，生成的shape为（14，288），意思是在同一时刻对所有路段在那一时刻的速度求平均。
    # 不应该这样求平均，这样平均的结果比较差，应该求一个路段，在一天内的所有非0元素的平均。
    # speed_data_mean = tf.math.reduce_mean(speed_data, axis=-1)
    # 这样会把0算进去，求出来的平均值会偏小
    # speed_data_mean = tf.math.reduce_mean(speed_data, axis=1)       # 这种计算不准确

    # 把平局速度的值复制多份，用于在tf。where上取值

    new_speed_data = tf.expand_dims(new_speed_mean, axis=-1)     # shape(14,288,4248,1)
    X_in_1 = tf.concat([new_speed_data,X_in[...,1:]], axis=-1)
    # 平均值填入空缺处

    # 数据标准化处理
    # # m = tf.concat([tf.ones(speed_data.shape[:-1]+[1]),tf.zeros(speed_data.shape[:-1]+(speed_data.shape[-1]-1))],axis=-1)
    # mean = tf.math.reduce_mean(speed_data)
    # std = tf.math.reduce_std(speed_data)
    # speed_data = tf.math.divide(tf.math.subtract(speed_data, mean), std)
    # # X_in = tf.where(m,tf.tile(speed_data,[1,1,1,X_in.shape[-1]-1]),X_in)
    # X_in_1 = tf.concat([tf.expand_dims(speed_data, axis=-1), X_in[..., 1:]], axis=-1)

    tgat_output = tgat(X_in_1)
    # print(tgat_output.shape)
    output = []
    tgat_output = tf.transpose(tgat_output, [0, 2, 1, 3])       # 换完之后是（14，4248，288，F_）

    # 被添加完之后再换轴
    new_speed_data = tf.transpose(new_speed_data, [0,2,1,3])
    for i in range(tgat_output.shape[0]):
        # time_feature_embedding = dense4time_feature(time_feature[i])
        # mix_feature = tf.concat([new_speed_data[i], tgat_output[i]], axis=-1)
        bg = bigru(tgat_output[i])
        con = conv1d(tgat_output[i])

        # bg = bigru(mix_feature)
        # con = conv1d(mix_feature)
        # con2 = conv1d_2(con)
        cc = concat([new_speed_data[i],bg, con])     # 在此处拼接一个平均速度数据，效果并不好，所以去掉，当然也有可能是速度求完平局后不太对，没去掉非零元素。
        # drop = Dropout(0.02)(cc)                    # 加个dropout层看看
        den = dense(cc)
        den2 = dense2(den)
        den3 = dense3(den2)

        output.append(den3)
    ret = tf.stack(output, 0)  # ret's shape is (288,4248,1)
    # # 数据反标准化
    # ot = tf.math.add(tf.math.multiply(ret, std), mean)

    ret = tf.transpose(ret, [0, 2, 1, 3])   # 维度调换
    # ret = ret * tf.expand_dims(tf.cast(speed_data, tf.float32),axis=-1)
    model = Model(inputs=X_in, outputs=ret)

    # optimizer = Adam(learning_rate=optimizer_lr)
    # model.compile(optimizer=optimizer,
    #               loss='mean_squared_error',  # 'mean_squared_error', # 如果是输入多天的数据，loss就不好用自定义的了
    #               weighted_metrics=['acc'],
    #               )

    return model


def get_model3(F_=7,
               gru_output_dim=12,
               conv1d_filters=12,
               conv1d_kernel_size=7,
               conv1d_strides=1,
               optimizer_lr=0.01,
               gat_heads=2,
               bigru_merge_mode='concat'):  # decoder改用全连接

    conv1d_padding = 'same'

    X_in = Input(shape=(288, 4248, 13), batch_size=1)  # batch_size指定时才可以在后面的程序中对第一维进行遍历
    forward_layer = GRU(gru_output_dim, return_sequences=True)
    backward_layer = GRU(gru_output_dim, return_sequences=True, go_backwards=True)
    bigru = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode=bigru_merge_mode)

    conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                    strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
    conv1d_2 = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                      strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
    concat = Concatenate(axis=2)  # ([x, y]), axis=2表示将第三维合并
    dense = Dense(units=32, activation='softmax')
    dense2 = Dense(units=16, activation='softmax')
    dense3 = Dense(units=1)
    # 对于简单、无状态的自定义操作，你也许可以通过 layers.core.Lambda 层来实现。即使用keras.core.lambda()
    tgat = Timeserial_GAT(F_, adj, gat_heads)

    # 数据标准化处理
    # speed_data = X_in[:, :, :, 1]
    # # m = tf.concat([tf.ones(speed_data.shape[:-1]+[1]),tf.zeros(speed_data.shape[:-1]+(speed_data.shape[-1]-1))],axis=-1)
    # mean = tf.math.reduce_mean(speed_data)
    # std = tf.math.reduce_std(speed_data)
    # speed_data = tf.math.divide(tf.math.subtract(speed_data, mean), std)
    # # X_in = tf.where(m,tf.tile(speed_data,[1,1,1,X_in.shape[-1]-1]),X_in)
    # X_in_1 = tf.concat([tf.expand_dims(speed_data, axis=-1), X_in[..., 1:]], axis=-1)

    tgat_output = tgat(X_in)
    # print(tgat_output.shape)
    output = []
    tgat_output = tf.transpose(tgat_output, [0, 2, 1, 3])
    conv_bigru_timedistributed = TimeDistributed(Conv_Bigru(gru_output_dim=gru_output_dim,
                                                            conv1d_filters=conv1d_filters,
                                                            conv1d_kernel_size=conv1d_kernel_size,
                                                            conv1d_strides=conv1d_strides,
                                                            conv1d_padding=conv1d_padding),
                                                 input_shape=tgat_output.shape[-2:])(
        tgat_output)
    tmp_list = []
    for i in range(conv_bigru_timedistributed.shape[0]):
        den_out = dense(conv_bigru_timedistributed[i])
        den_out2 = dense2(den_out)
        den_out3 = dense3(den_out2)
        tmp_list.append(den_out3)
    ret = tf.stack(tmp_list)
    # dense_timedistributed = TimeDistributed(dense, input_shape=conv_bigru_timedistributed.shape[-2:])(conv_bigru_timedistributed)
    # dense2_timedistributed = TimeDistributed(dense2, input_shape=dense_timedistributed.shape[-2:])(dense_timedistributed)
    # dense3_timedistributed = TimeDistributed(dense3, input_shape=dense2_timedistributed.shape[-2:])(dense2_timedistributed)
    # ot = tf.transpose(dense3_timedistributed, [0, 2, 1, 3])
    # for i in range(tgat_output.shape[0]):
    #     # for j in range(tgat_output.shape[1])
    #     bg = bigru(tgat_output[i])
    #     con = conv1d(tgat_output[i])
    #     con2 = conv1d_2(con)
    #     cc = concat([bg, con2])
    # dec = decoder_bigru(cc)

    # denlist=[]
    # for j in range(tgat_output.shape[1]):
    #     den = dense(cc[j])
    #     den2 = dense2(den)
    #     den3 = dense3(den2)
    #     denlist.append(den3)
    # den3ret = tf.stack(denlist, axis=0)

    # 根据实验，Dense层只对最后一维做全连接，全连接层的kernel矩阵大小为输入张量的最后一维*设置的输出维度大小
    # 所以向上面注释掉的那种想法是不正确的。
    # den = dense(cc)  # 这个可以试着减少参数，用堆叠的方式拼起来
    # den2 = dense2(den)
    # den3 = dense3(den2)
    #
    # output.append(den3)
    # ret = tf.stack(output, 0)  # ret's shape is (288,4248,1)
    # # 数据反标准化
    # ot = tf.math.add(tf.math.multiply(ret, std), mean)

    ot = tf.transpose(ret, [0, 2, 1, 3])  # 维度调换

    model = Model(inputs=X_in, outputs=ot)

    optimizer = Adam(learning_rate=optimizer_lr)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',  # 'mean_squared_error', # 如果是输入多天的数据，loss就不好用自定义的了
                  weighted_metrics=['acc'],
                  )

    return model


#
# class Loss_masked(tf.keras.losses.Loss):
#     def __init__(self, mask, **kwargs):
#         self.mask = tf.expand_dims(mask, 0)
#         super(Loss_masked, self).__init__(**kwargs)
#
#     def call(self, y_true, y_pred):
#         # print(self.mask.shape,y_true.shape,y_pred.shape)
#         # assert self.mask.shape==y_true.shape==y_pred.shape
#         # y_pred = tf.convert_to_tensor_v2(y_pred)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         return tf.math.reduce_mean(
#             tf.math.square(tf.math.subtract(tf.where(self.mask, y_pred, 0), tf.where(self.mask, y_true, 0))), axis=-1)


def train_model(data_name, feature_name):
    learning_rate = 0.00001  # Learning rate for Adam
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 49.795909892694354,rmse: 5.5290348272584255

    # N = true_feature.shape[1]
    es_patience = 100

    n_attn_heads = 1
    dropout_rate = 0.4
    l2_reg = 5e-4 / 2
    model = get_model2()
    # loss_masked = Loss_masked(train_mask)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',  #my_loss,#  'mean_squared_error', # 如果是输入多天的数据，loss就不好用自定义的了
                  weighted_metrics=['acc'],
                  )

    model.summary()
    # Callbacks
    # es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
    tb_callback = TensorBoard(batch_size=1,log_dir='./logs')

    # ****************************************************************
    # 这里要记得改，切记
    # mc_callback = ModelCheckpoint('./logs/best_model3.h5',
    #                               monitor='loss',  # 选择监测的参数，loss，acc，val_loss，val_acc
    #                               save_best_only=True,
    #                               save_weights_only=True,  # False时不仅保存参数，也保存模型，会运行更慢。
    #                               mode='min',
    #                               save_freq='epoch')  # 'epoch'或者是一个整数
    # model.build(None)
    # model.summary()

    # train_speed_data = np.load("./data/{}".format(data_name))['arr_0']
    # train_speed_mask = np.load("./data/{}".format(data_name))['arr_1']
    # train_speed_data = np.load("./data/all_train_data20random_missing.npz")['arr_0']
    # validation_data = (tf.expand_dims(true_feature, 0),
    #                    tf.expand_dims(speed_feature, 0),
    #                    )
    # model.load_weights('./logs/best_model3.h5')
    all_feature = np.load(r'./data/all_feature.npy')        # (14,288,4248,13)
    # 对训练数据单独进行scale，但是StandardScaler类只能对二维数据进行处理，因此要先reshape一下，不知道这样做对不对。
    scaler4all_feature = StandardScaler()
    scaler4true_all_speed_data = StandardScaler()

    all_feature_mat = np.reshape(all_feature, [-1, 13])
    all_feature_mat_scaled = scaler4all_feature.fit_transform(all_feature_mat)
    all_feature_scaled = np.reshape(all_feature_mat_scaled, [14, 288, 4248, 13])
    true_all_data = np.load('./all_speed_data.npz')
    # shape(14,288,4248,1)
    true_all_speed_data, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    true_all_speed_data_mat = np.reshape(true_all_speed_data, [-1, 1])
    true_all_speed_data_mat_scaled = scaler4true_all_speed_data.fit_transform(true_all_speed_data_mat)
    true_all_speed_data_scaled = np.reshape(true_all_speed_data_mat_scaled, [14,288,4248,1])
    # 尝试对速度这一维度进行scale，
    # scaler = StandardScaler()
    # all_feature_scaled = scaler.fit_transform(np.expand_dims(np.reshape(all_feature[..., 0], [-1]), axis=-1))
    # true_all_speed_data_scaled = scaler.transform(true_all_speed_data)
    epochs = 100
    model.fit(all_feature_scaled[:10],
              true_all_speed_data[:10],
              # true_all_speed_data_scaled[:10],
              epochs=epochs,  # epochs 300
              batch_size=2,
              # validation_data=validation_data,
              shuffle=True,  # Shuffling data means shuffling the whole graph
              # callbacks=[mc_callback, tb_callback],  # callbacks=[es_callback, tb_callback, mc_callback]
              )

    # my model use data all_train_data20random_missing.npz epochs: 10？ mape: 42.27985794981111,rmse: 4.019087635217096
    # my model use data all_train_data20random_missing.npz epochs: 30 mape: 49.730531022205994,rmse: 5.968522419635606
    # my model use data all_train_data20random_missing.npz epochs: 10! mape: 61.87384123009226,rmse: 7.135004542648857
    # my model use data all_train_data20random_missing.npz epochs: 40 mape: 48.884355610238856,rmse: 5.88698438321226
    # x，y分别做scale后的10轮结果。lr：0.01  epochs: 10 mape: 103.19824074635093,rmse: 10.01700903887512
    # lr:0.000001 epochs: 4 mape: 56.95077387424444,rmse: 6.654440045946629
    # lr=0.00001 epochs: 10 mape: 45.19785299272371,rmse: 4.313149385966981
    # my model lr 0.01 epochs: 4 mape: 49.19702917989168,rmse: 5.101876698674287
    # my model use data all_train_data20random_missing.npz epochs: 40 mape: 52.69890934129867,rmse: 6.0619099168303485
    # my model lr 0.0001 epochs: 4 mape: 94.03413238408373,rmse: 9.439349817034703
    # my model lr 0.01 epochs: 4 mape: 49.19702917989168,rmse: 5.101876698674287
    # my model lr 0.01 epochs: 60 mape: 100.88957119137085,rmse: 9.904628514410149 光看loss，在40轮的时候就停止下降了
    # my model lr 1e-05 epochs: 40 mape: 101.05770881398549,rmse: 9.96960137361105
    # my model lr 1e-05 epochs: 4 mape: 101.74235256103307,rmse: 9.877179556237548
    # my model lr 0.1 epochs: 4 mape: 55.5339305898174,rmse: 5.599063056904649
    # my model lr 0.1 epochs: 12 mape: 90.60247809652043,rmse: 8.935796963148807
    # my model lr 0.1 epochs: 7 mape: 56.93645927159993,rmse: 6.414716932305194
    # 不用eval_model函数时进行评估模型好坏的代码
    # train_feature = np.load(r'./data/{}'.format(feature_name))
    train_feature = np.load('./data/{}'.format(feature_name))
    true_all_speed_data = np.squeeze(true_all_speed_data)
    true_all_mask = np.squeeze(true_all_mask)
    train_data = np.load("./data/{}".format(data_name))
    train_mask = train_data['arr_1']        # train_mask shepe (14,288,4248)
    mask = true_all_mask ^ train_mask
    pre = model.predict(np.reshape(scaler4all_feature.transform(np.reshape(train_feature[10:],[-1, 13])),[-1,288,4248,13]))
    pre = np.squeeze(pre)
    # pre_scaled = model.predict(np.reshape(scaler4all_feature.transform(np.reshape(train_feature[10:],[-1, 13])),[-1,288,4248,13])
    # pre = np.reshape(scaler4true_all_speed_data.inverse_transform(np.reshape(pre_scaled,[-1,1])),[-1,288,4248])
    deviation_true_data, deviation_pret_data = get_deviation(true_all_speed_data[10:], true_all_mask[10:], pre)
    print("deviation_pret_data", deviation_pret_data)
    print("deviation_true_data", deviation_true_data)
    a2 = true_all_speed_data[10:]
    m2 = mask[10:]
    mape_res20 = mape(a2[m2], pre[m2])
    rmse_res20 = rmse(a2[m2], pre[m2])
    # mape_res20 = mape(true_all_speed_data[mask], pre[mask])
    # rmse_res20 = rmse(true_all_speed_data[mask], pre[mask])
    return_str = 'my model lr {} epochs: {} mape: {},rmse: {}'.format(learning_rate, epochs, mape_res20, rmse_res20)
    print(return_str)
    with open('train_my_model_res', "a+") as f:
        f.write(return_str + '\n')


    # eval_model(data_name, feature_name, model, epochs)


    # model.save_weights(filepath='./logs/best_model.h5')
    # model.fit([tf.random.normal((1,288,4248,11)),tf.where(tf.random.normal((1,4248,4248))<0.9,1,0)],
    #           tf.random.normal((1,288,4248,1)))

    # model.load_weights('./logs/best_model.h5')
    #
    # # Evaluate model
    # eval_results = model.evaluate([true_feature,adj_mat],
    #
    #                               batch_size=1,
    #                               verbose=0)


# def eval_model(data_name, feature_name, model=None, epochs=None):
#     if model is None:
#         model = get_model2()
#         model.load_weights('./logs/best_model3.h5')
#
#     # 获得包含特征的张量，（14，288，4248，11）
#     train_feature = np.load(r'./data/{}'.format(feature_name))
#
#     # 获得能得到的最完整的速度张量和它的掩码，（14，288，4248）
#     true_all_data = np.load('./all_speed_data.npz')
#     # (14,288,4248,1)
#     true_all_speed_data, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
#     true_all_speed_data = np.squeeze(true_all_speed_data)
#     true_all_mask = np.squeeze(true_all_mask)
#
#     # 获得训练时的速度张量和掩码，（14，288，4248）
#     train_data = np.load("./data/{}".format(data_name))
#     # train_speed_feature = train_data['arr_0']
#     train_mask = train_data['arr_1']
#     mask = true_all_mask ^ train_mask
#
#     pre = model.predict(train_feature[10:])
#     pre = np.squeeze(pre)
#     deviation_true_data, deviation_pret_data = get_deviation(true_all_speed_data[10:], true_all_mask[10:], pre)
#     print("deviation_pret_data", deviation_pret_data)
#     print("deviation_true_data", deviation_true_data)
#     a2 = true_all_speed_data[10:]
#     m2 = mask[10:]
#     mape_res20 = mape(a2[m2], pre[m2])
#     rmse_res20 = rmse(a2[m2], pre[m2])
#     return_str = 'my model use data {} epochs: {} mape: {},rmse: {}'.format(data_name, epochs, mape_res20, rmse_res20)
#     print(return_str)
#     with open('train_my_model_res', "a+") as f:
#         f.write(return_str + '\n')
#     # my model missing 20 mape: 54.89880543062464,rmse: 6.057448383899675
#     # my model missing 20 mape: 50.85719952002699,rmse: 5.701436543350604


if __name__ == "__main__":
    # support = tf.sparse.SparseTensor(indices=adj[0],values=adj[1],dense_shape=adj[2])
    # print(type(support))
    # print('*****')
    # print(support.shape)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_name', default="xx", type=str, help="xxq")
    # parser.add_argument('--feature_name', default="xx", type=str, help="xxq")
    # parser.add_argument('--epochs', default=25, type=int, help='train epochs')
    # args = parser.parse_args()
    # train_model(args.data_name,args.feature_name)

    train_model("all_train_data20random_missing.npz", "train_feature20random_missing.npy")
    # eval_model("all_train_data20random_missing.npz", "train_feature20random_missing.npy")

    # train_model("all_train_data20no_random_missing_road.npz", "train_feature20no_random_missing_road.npy")
    # train_model("all_train_data20no_random_missing_time.npz", "train_feature20no_random_missing_time.npy")
    # train_model("all_train_data40random_missing.npz", "train_feature40random_missing.npy")
    # train_model("all_train_data40no_random_missing_road.npz", "train_feature40no_random_missing_road.npy")
    # train_model("all_train_data40no_random_missing_time.npz", "train_feature40no_random_missing_time.npy")
    # train_model("all_train_data60random_missing.npz", "train_feature60random_missing.npy")
    # train_model("all_train_data60no_random_missing_road.npz", "train_feature60no_random_missing_road.npy")
    # train_model("all_train_data60no_random_missing_time.npz", "train_feature60no_random_missing_time.npy")
    # train_model("all_train_data80random_missing.npz", "train_feature80random_missing.npy",epochs=5)
    # train_model("all_train_data80no_random_missing_road.npz", "train_feature80no_random_missing_road.npy")
    # train_model("all_train_data80no_random_missing_time.npz", "train_feature80no_random_missing_time.npy")

    # eval_model()

    # my model use data all_train_data20random_missing.npz mape: 46.22000420869335,rmse: 5.250959928153677
    # my model use data all_train_data40random_missing.npz mape: 75.47186711415502,rmse: 8.033122982530895
    # my model use data all_train_data60random_missing.npz epochs: 10 mape: 75.31992423615912,rmse: 8.43591866581605
    # my model use data all_train_data80random_missing.npz epochs: 10 mape: 90.22946952359315,rmse: 9.511333869257182
    # my model use data all_train_data80random_missing.npz epochs: 5 mape: 91.28546573618716,rmse: 9.64331394746252

    # my model use data all_train_data20no_random_missing_road.npz epochs: 10 mape: 66.75901266111043,rmse: 7.449601522914078
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 71.20064865152233,rmse: 7.393288687209375
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 43.015797397925766,rmse: 4.9121722619730335
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 45.93785349379076,rmse: 4.790914256481592
    # 改用loss函数后的最好效果，训练20轮，解码层dense激活函数relu，
    # my model use data all_train_data20random_missing.npz epochs: 20 mape: 31.430187086511424,rmse: 2.3641022064507777
    # 多训练了10轮，效果没什么提升
    # my model use data all_train_data20random_missing.npz epochs: 30 mape: 30.785256082992586,rmse: 2.3388215228610436
    # 如果只训练10轮，效果这样
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 33.50785121133132,rmse: 3.0188408990606788
    # 去掉了时间特征的嵌入，lr=0.02，解码层dense激活函数是relu,把用平均值补全后的速度特征在gat之后又拼接了一次。loss换成my_loss
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 32.16923988747752,rmse: 2.473830844790384
    # 同样训练10轮，dense改用softmax激活函数后效果更差了，最后一轮的loss为loss: 1.9585 - acc: 0.2038
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 35.306259192382356,rmse: 3.479958343825557
    # 不用激活函数的结果和用softmax的结果差不多，说明使用relu应该还是有用的
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 40.08759364113953,rmse: 3.604059966080607
    # 减少一层dense，并恢复使用relu试试，结果不怎么样，似乎不如三层dense解码效果好
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 36.81153756186533,rmse: 3.471115680050633
    # 把用平均速度填充过的速度数据和编码器的输出拼接后，效果也不怎么样
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 44.223809116565455,rmse: 5.077333884644108
    # 设置和上面一样，奇怪的是训练轮数少了，效果比轮数多的还好一点。
    # my model use data all_train_data20random_missing.npz epochs: 4 mape: 45.35817252324353,rmse: 3.9280771974456785
    # 把bigru的俩个gru换成了一个gru，lr=0.02时，三十轮的效果，平均速度没填在解码器之前
    # my model use data all_train_data20random_missing.npz epochs: 30 mape: 32.55522746515484,rmse: 2.288811352799023
    #
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 31.451755653859127,rmse: 2.3219439768034786
    # 把在解码器前拼接的速度数据去掉，同样训练十轮对比一下，结果差一些，残差还是有用
    # my model use data all_train_data20random_missing.npz epochs: 10 mape: 33.76116485569354,rmse: 2.9281050194341467
    # 加了dropout层，
    # my model use data all_train_data20random_missing.npz epochs: 30 mape: 31.723721796246313,rmse: 2.3387883634707722


