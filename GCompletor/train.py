import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Conv1D, Concatenate
from tensorflow.keras.layers import GRU, Input, Dense
from tensorflow.keras.optimizers import Adam

from graph_attention_layer import Timeserial_GAT
from utils import get_roadId_list_and_adj_mat


# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]  # 用gpu跑，并且动态分配内存
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 用gpu跑内存不够，用cpu可以跑
# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# shape为（288，4248，11）,(288,4248,1)
# true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = get_feature(0.2)
true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = np.load(
    './feature.npz').values()
# 在数据本身就缺失的情况下，再丢掉一部分，丢掉的比例是多少？  应该是再丢掉20%
_, adj_mat = get_roadId_list_and_adj_mat()

learning_rate = 0.03  # Learning rate for Adam

epochs = 10000
N = true_feature.shape[1]
es_patience = 100
F_ = 7
n_attn_heads = 1
dropout_rate = 0.4
l2_reg = 5e-4 / 2
gru_output_dim = 12
conv1d_filters = 12
conv1d_kernel_size = 7
conv1d_strides = 1
conv1d_padding = 'same'

X_in = Input(shape=(288, 4248, 11), batch_size=1)  # batch_size指定时才可以在后面的程序中对第一维进行遍历
A_in = Input(shape=(4248, 4248), batch_size=1)

# graph_attention_1 = GraphAttention(F_,
#                                    attn_heads=n_attn_heads,
#                                    attn_heads_reduction='concat',
#                                    dropout_rate=dropout_rate,
#                                    activation='elu',
#                                    kernel_regularizer=l2(l2_reg),
#                                    attn_kernel_regularizer=l2(l2_reg))
# graph_attention_2 = GraphAttention(F_,
#                                    attn_heads=n_attn_heads,
#                                    attn_heads_reduction='concat',
#                                    dropout_rate=dropout_rate,
#                                    activation='elu',
#                                    kernel_regularizer=l2(l2_reg),
#                                    attn_kernel_regularizer=l2(l2_reg))
forward_layer = GRU(gru_output_dim, return_sequences=True)
backward_layer = GRU(gru_output_dim, return_sequences=True, go_backwards=True)
bigru = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='concat')

conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
conv1d_2 = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                  strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]
concat = Concatenate(axis=2)  # ([x, y]), axis=2表示将第三维合并

# decoder
forward_layer2 = GRU(32, return_sequences=True)  # 这里好像没法指定输出维度和状态维度不相同
backward_layer2 = GRU(32, return_sequences=True, go_backwards=True)
decoder_bigru = Bidirectional(forward_layer2, backward_layer=backward_layer2, merge_mode='concat')
dense = Dense(units=1)

# 对于简单、无状态的自定义操作，你也许可以通过 layers.core.Lambda 层来实现。即使用keras.core.lambda()
tgat = Timeserial_GAT(F_)

# 数据标准化处理
speed_data = X_in[:,:,:,1]
# m = tf.concat([tf.ones(speed_data.shape[:-1]+[1]),tf.zeros(speed_data.shape[:-1]+(speed_data.shape[-1]-1))],axis=-1)
mean = tf.math.reduce_mean(speed_data)
std = tf.math.reduce_std(speed_data)
speed_data = tf.math.divide(tf.math.subtract(speed_data, mean), std)
# X_in = tf.where(m,tf.tile(speed_data,[1,1,1,X_in.shape[-1]-1]),X_in)
X_in_1=tf.concat([tf.expand_dims(speed_data,axis=-1),X_in[...,1:]],axis=-1)


tgat_output = tgat([X_in_1, A_in])
# print(tgat_output.shape)
output = []
tgat_output = tf.transpose(tgat_output, [0, 2, 1, 3])
for i in range(tgat_output.shape[0]):
    # for j in range(tgat_output.shape[1])
    bg = bigru(tgat_output[i])
    con = conv1d(tgat_output[i])
    con2 = conv1d_2(con)
    cc = concat([bg, con2])
    dec = decoder_bigru(cc)
    den = dense(dec)
    output.append(den)
ret = tf.stack(output, 0)  # ret's shape is (288,4248,1)
# bg=bigru(sta)
# con=conv1d(sta)
# cc=concat([bg,con])
# dec=decoder_bigru(cc)               # shape (4248,288,1)得换轴
#

# 数据反标准化
ot = tf.math.add(tf.math.multiply(ret, std), mean)

ot = tf.transpose(ot, [0, 2, 1, 3])  # 维度调换

model = Model(inputs=[X_in, A_in], outputs=ot)


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

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss= loss_masked,        # 'mean_squared_error' ,
              weighted_metrics=['acc'],
              )
model.summary()
# Callbacks
# es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
# tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint('./logs/best_model.h5',
                              monitor='val_loss',    # 选择监测的参数，loss，acc，val_loss，val_acc
                              save_best_only=True,
                              save_weights_only=True,  # False时不仅保存参数，也保存模型，会运行更慢。
                              save_freq='epoch')        # 'epoch'或者是一个整数
# model.build(None)
# model.summary()

validation_data = ([tf.expand_dims(true_feature, 0), tf.expand_dims(adj_mat, 0)],
                   tf.expand_dims(speed_feature, 0),
                   )
model.load_weights('./logs/best_model.h5')

model.fit([tf.expand_dims(train_feature, 0), tf.expand_dims(adj_mat, 0)],
          tf.expand_dims(train_speed_feature, 0),
          epochs=20,  # epochs
          batch_size=1,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[mc_callback],  # callbacks=[es_callback, tb_callback, mc_callback]
          workers=0)
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
