import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Concatenate, Dense, Add
from baselines.metrics import mape,rmse
import os

from utils import get_deviation

os.environ['CUDA_VISIBLE_DEVICES']='1'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental_run_functions_eagerly(True)


class Loss_masked(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(Loss_masked, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        self.mask=y_true!=0
        # print(self.mask.shape,y_true.shape,y_pred.shape)
        # assert self.mask.shape==y_true.shape==y_pred.shape
        # y_pred = tf.convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.math.reduce_mean(
            tf.math.square(tf.math.subtract(tf.where(self.mask, y_pred, 0), tf.where(self.mask, y_true, 0))), axis=-1)

def get_model():
    window_size = 288
    sensor = 323
    cnn1_filter = 16
    cnn2_filter = 16
    cnn3_filter = 16
    cnn1_kernel_size = 1
    cnn2_kernel_size = 3
    cnn3_kernel_size = 5
    encoder_lstm1_units = 32
    encoder_lstm2_units = 32
    # decoder_lstm1_units = 16
    # decoder_lstm2_units = 16

    input_shape = ( window_size,sensor)
    inp = Input(shape=input_shape)

    # filters: Integer, the dimensionality of the output space
    # kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    conv1_kernel1 = Conv1D(filters=cnn1_filter, kernel_size=cnn1_kernel_size, strides=1, padding='same')(inp)
    conv1_kernel2 = Conv1D(filters=cnn2_filter, kernel_size=cnn2_kernel_size, strides=1, padding='same')(inp)
    conv1_kernel3 = Conv1D(filters=cnn3_filter, kernel_size=cnn3_kernel_size, strides=1, padding='same')(inp)
    conca = Concatenate(axis=-1)([conv1_kernel1, conv1_kernel2, conv1_kernel3])

    # 论文原文
    # It has two sets of LSTM cells which propagate states in two opposite directions.
    # _,lstm1_final_memory_state,lstm1_final_carry_state = LSTM(units=encoder_lstm1_units, return_state=True)(conca)
    # _,lstm2_final_memory_state,lstm2_final_carry_state = LSTM(units=encoder_lstm2_units,
    #                                                           go_backwards=True,return_state=True)(conca)

    # 不懂作者是怎么利用最后的状态的，因此这里做一些改动
    lstm1 = LSTM(units=encoder_lstm1_units, return_sequences=True)
    lstm2 = LSTM(units=encoder_lstm2_units, return_sequences=True, go_backwards=True)

    # 感觉论文中应该说是用了两个，而非双向，因为两个的h和c没有关联。
    encoder_bilstm = Bidirectional(layer=lstm1, backward_layer=lstm2, merge_mode='concat')(conca)

    # 这里因为要使用残差，需要对decoder的bilstm的输出维度进行特别指定。
    decoder_lstm1_units = (cnn1_filter + cnn2_filter + cnn3_filter) // 2
    decoder_lstm2_units = (cnn1_filter + cnn2_filter + cnn3_filter) // 2
    decoder_lstm1 = LSTM(units=decoder_lstm1_units, return_sequences=True)
    decoder_lstm2 = LSTM(units=decoder_lstm2_units, return_sequences=True, go_backwards=True)
    decoder_bilstm = Bidirectional(layer=decoder_lstm1, backward_layer=decoder_lstm2, merge_mode='concat')(
        encoder_bilstm)
    residual = Add()([conca, decoder_bilstm])
    den_ret1 = Dense(units=1000,activation='relu')(residual)
    # den_ret2 = Dense(units=2400)(den_ret1)
    output = Dense(units=sensor)(den_ret1)
    model = Model(inputs=inp, outputs=output)
    # model.summary()
    return model

def gen_data():
    window_size = 288
    data_name_list = ["all_train_data20random_missing",
                      "all_train_data40random_missing",
                      "all_train_data60random_missing",
                      "all_train_data80random_missing",
                      "all_train_data20no_random_missing_time",
                      "all_train_data40no_random_missing_time",
                      "all_train_data60no_random_missing_time",
                      "all_train_data80no_random_missing_time",
                      "all_train_data20no_random_missing_road",
                      "all_train_data40no_random_missing_road",
                      "all_train_data60no_random_missing_road",
                      "all_train_data80no_random_missing_road"]
    for data_name in data_name_list:
        data_miss = np.load(os.path.join("./data",data_name+'.npz'))
        train_speed_data_miss, train_mask_miss = data_miss['arr_0'], data_miss['arr_1']
        day, time, sensor = train_speed_data_miss.shape
        ret = []
        for d in range(day):
            li = []
            for i in range(time-window_size):
                li.append(train_speed_data_miss[d,i:window_size+i])
            data = np.stack(li)
            ret.append(data)
        ret = np.stack(ret)

        np.save('./data/cnnbilstmautoencoder_{}'.format(data_name), ret)
def train_model(epochs = 50):
    # window_size = 100  # 每次训练时的窗口大小
    shift = 1  # 每次采样的偏移度
    stride = 1  # 步长


    batch_size = 16
    shuffle = True
    # 准备数据
    # true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = np.load(
    #     './feature.npz').values()
    data_path='./data/seattle/random_missing/seattle_random_missing20.npy'
    # data_path='./data/seattle/random_missing/seattle_random_missing40.npy'
    # data_path='./data/seattle/random_missing/seattle_random_missing60.npy'
    # data_path='./data/seattle/random_missing/seattle_random_missing80.npy'
    # data_path='./data/seattle/no_random_missing_time/seattle_no_random_missing_time20.npy'
    # data_path='./data/seattle/no_random_missing_time/seattle_no_random_missing_time40.npy'
    # data_path='./data/seattle/no_random_missing_time/seattle_no_random_missing_time60.npy'
    # data_path='./data/seattle/no_random_missing_time/seattle_no_random_missing_time80.npy'
    # data_path ='./data/seattle/no_random_missing_road/seattle_no_random_missing_road20.npy'
    # data_path ='./data/seattle/no_random_missing_road/seattle_no_random_missing_road40.npy'
    # data_path ='./data/seattle/no_random_missing_road/seattle_no_random_missing_road60.npy'
    # data_path ='./data/seattle/no_random_missing_road/seattle_no_random_missing_road80.npy'


    train_speed_feature = np.load(data_path).reshape([365,288,323])
    # train_speed_feature = np.reshape(train_speed_feature,(-1,100,4248))
    # squeeze之后shape为（288，4248）
    # train_speed_feature = tf.squeeze(train_speed_feature)
    # dataset = tf.data.Dataset.from_tensor_slices(train_speed_feature).window(window_size, shift=shift, stride=stride,
    #                                                                          drop_remainder=True)
    # dataset = dataset.map(tf.stack)

    lr = 0.01

    model = get_model()
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss=Loss_masked(),#'mean_squared_error',  # 'mean_squared_error',
                  weighted_metrics=['acc'],
                  )
    model.summary()

    mc_callback = ModelCheckpoint('./logs/best_cnn_bilstm_autoencoder.h5',
                                  monitor='loss',  # 选择监测的参数，loss，acc，val_loss，val_acc
                                  save_best_only=True,
                                  save_weights_only=True,  # False时不仅保存参数，也保存模型，会运行更慢。
                                  save_freq='epoch')

    # model.load_weights('./logs/best_cnn_bilstm_autoencoder.h5')

    model.fit(x=train_speed_feature,
              y=train_speed_feature,
              epochs=80,#epochs,  # epochs 300
              batch_size=batch_size,
              shuffle=shuffle,  # Shuffling data means shuffling the whole graph
              # callbacks=[mc_callback]
              )

    # eval
    pret = model.predict(train_speed_feature)
    true_data=np.array(pd.read_csv('./data/seattle/Speed2.csv').values).reshape([365,288,323])
    mask=train_speed_feature==0
    TRTF_res2_mape2 = mape(true_data[mask], pret[mask])
    TRTF_res2_rmse2 = rmse(true_data[mask], pret[mask])

    ret_str = "ae_data:{}mape={}\n".format(data_path, TRTF_res2_mape2)
    ret_str += "ae_missing_rate{}rmse={}".format(data_path, TRTF_res2_rmse2)
    print(ret_str)

    # true_mask_ten = train_speed_feature != 0
    # deviation_true_data, deviation_pret_data = get_deviation(true_data, true_mask_ten, pret)
    # print("ae_deviation_pret_data", deviation_pret_data)
    # print("deviation_true_data", deviation_true_data)
    #
    # return ret_str

    # eval_model(model,epochs)

def eval_model(data_name,model=None,epochs=None):
    # true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = np.load(
    #     './feature.npz').values()
    train_speed_data = np.load(os.path.join('./data',data_name+".npz"))
    # speed_feature = speed_data['arr_0']    #  (14,288,4248)
    train_mask = train_speed_data['arr_1']
    speed_data = np.load('./all_speed_data.npz')
    speed_feature = np.squeeze(speed_data['arr_0'])
    true_mask = np.squeeze(speed_data['arr_1'])


    # tmp = np.zeros(speed_feature.shape)
    if model is None:

        model = get_model()
        model.load_weights('./logs/best_cnn_bilstm_autoencoder.h5')

    train_speed_feature = np.load(os.path.join('./data', "cnnbilstmautoencoder_"+data_name+'.npy')) # shape(14,188,100,4248)
    # train_speed_feature = np.reshape(train_speed_feature,(-1,100,4248))
    data_shape = train_speed_feature.shape   # (14,188,100,4248)
    res_list=[]
    for d in range(data_shape[0]):
        pret = model.predict(train_speed_feature[d])
        tmp = np.zeros((288, 4248))

        for i in range(data_shape[1]):
            # pret = model.predict(train_speed_feature[i])
            tmp[i:data_shape[2]+i] += pret[i]

        for i in range(data_shape[2]-1):
            tmp[i]=tmp[i]/(i+1)
            tmp[tmp.shape[0]-i-1] = tmp[tmp.shape[0]-i-1]/(i+1)
        tmp[data_shape[2]-1:tmp.shape[0]-data_shape[2]+1] = tmp[data_shape[2]-1:tmp.shape[0]-data_shape[2]+1]/data_shape[2]
        res_list.append(tmp)
    all_pret = tf.stack(res_list)
    mask = true_mask ^ train_mask  # 两个掩码相异或，得到的才是在训练中没有，但我们又能得到的数据。

    deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask, all_pret)
    print("deviation_pret_data", deviation_pret_data)
    print("deviation_true_data", deviation_true_data)
    # mask = np.squeeze(mask, -1)

    # cnn_bilstm_ae_res2_mape2 = mape(speed_feature[mask], all_pret[mask])
    # cnn_bilstm_ae_res2_rmse2 = rmse(speed_feature[mask], all_pret[mask])
    # ret_str = "cnn_bilstm_ae_data_{}_epochs_{}_mape={}\n".format(data_name,epochs,cnn_bilstm_ae_res2_mape2)
    # ret_str += "cnn_bilstm_ae_data_{}_epochs_{}_rmse={}".format(data_name,epochs,cnn_bilstm_ae_res2_rmse2)
    # print(ret_str)

if __name__ == '__main__':
    train_model( )