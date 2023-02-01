import numpy as np
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
    sensor = 4248
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
    window_size = 100
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
    # data20miss = np.load("./data/all_train_data20random_missing.npz")
    # train_speed_data20miss, train_mask20miss = data20miss['arr_0'], data20miss['arr_1']
    # data40miss = np.load("./data/all_train_data40random_missing.npz")
    # train_speed_data40miss, train_mask40miss = data40miss['arr_0'], data40miss['arr_1']
    # data60miss = np.load("./data/all_train_data60random_missing.npz")
    # train_speed_data60miss, train_mask60miss = data60miss['arr_0'], data60miss['arr_1']
    # data80miss = np.load("./data/all_train_data80random_missing.npz")
    # train_speed_data80miss, train_mask80miss = data80miss['arr_0'], data80miss['arr_1']

    # true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = np.load(
    #     './feature.npz').values()
    # train_speed_feature = np.squeeze(train_speed_feature)

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
def train_model(epochs = 5):
    # window_size = 100  # 每次训练时的窗口大小
    shift = 1  # 每次采样的偏移度
    stride = 1  # 步长


    batch_size = 16
    shuffle = True
    # 准备数据
    # true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = np.load(
    #     './feature.npz').values()
    # train_speed_feature = np.load('./data/cnnbilstmautoencoder_{}.npy'.format(data_name))
    # train_speed_feature = np.reshape(train_speed_feature,(-1,100,4248))
    # squeeze之后shape为（288，4248）
    # train_speed_feature = tf.squeeze(train_speed_feature)
    # dataset = tf.data.Dataset.from_tensor_slices(train_speed_feature).window(window_size, shift=shift, stride=stride,
    #                                                                          drop_remainder=True)
    # dataset = dataset.map(tf.stack)
    # data_path='./data/all_train_data20random_missing.npz'
    # data_path='./data/all_train_data40random_missing.npz'
    # data_path='./data/all_train_data60random_missing.npz'
    # data_path='./data/all_train_data80random_missing.npz'
    # data_path = './data/all_train_data20no_random_missing_time.npz'
    # data_path = './data/all_train_data40no_random_missing_time.npz'
    # data_path = './data/all_train_data60no_random_missing_time.npz'
    # data_path = './data/all_train_data80no_random_missing_time.npz'
    # data_path='./data/all_train_data20no_random_missing_road.npz'
    # data_path='./data/all_train_data40no_random_missing_road.npz'
    # data_path='./data/all_train_data60no_random_missing_road.npz'
    data_path='./data/all_train_data80no_random_missing_road.npz'
    train_speed_feature_ = np.load(data_path)
    train_speed_feature ,train_mask= train_speed_feature_['arr_0'],train_speed_feature_['arr_1']
    lr = 0.003

    model = get_model()
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss=Loss_masked(),#'mean_squared_error',  # 'mean_squared_error',
                  # weighted_metrics=['acc'],
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
              epochs=200,#epochs,  # epochs 300
              batch_size=batch_size,
              shuffle=shuffle,  # Shuffling data means shuffling the whole graph
              # callbacks=[mc_callback]
              )

    # eval
    data = np.load(r'./all_speed_data.npz')
    speed_feature, mask = data['arr_0'], data['arr_1']
    speed_feature, mask = np.squeeze(speed_feature), np.squeeze(mask)  # 去掉最后一个维度,shape（14，288，4248）

    mask2=train_mask ^ mask
    pret = model.predict(train_speed_feature)
    ae_res2_mape2 = mape(speed_feature[mask2], pret[mask2])
    ae_res2_rmse2 = rmse(speed_feature[mask2], pret[mask2])

    ret_str = "ae_data:{}mape={}\n".format(data_path, ae_res2_mape2)
    ret_str += "ae_missing_rate{}rmse={}".format(data_path,ae_res2_rmse2)
    print(ret_str)
    # eval_model(data_name,model,epochs)

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

    cnn_bilstm_ae_res2_mape2 = mape(speed_feature[mask], all_pret[mask])
    cnn_bilstm_ae_res2_rmse2 = rmse(speed_feature[mask], all_pret[mask])
    ret_str = "cnn_bilstm_ae_data_{}_epochs_{}_mape={}\n".format(data_name,epochs,cnn_bilstm_ae_res2_mape2)
    ret_str += "cnn_bilstm_ae_data_{}_epochs_{}_rmse={}".format(data_name,epochs,cnn_bilstm_ae_res2_rmse2)
    print(ret_str)
    # cnn_bilstm_ae data all_train_data20random_missing epochs 1 mape:37.879737029308345, rmse:3.910190648814337
    # cnn_bilstm_ae data all_train_data20random_missing epochs 5 mape:35.072562345820074, rmse:3.621754929173983
    # cnn_bilstm_ae data all_train_data20random_missing epochs 5 mape:35.072562345820074, rmse:3.621754929173983
    # cnn_bilstm_ae data all_train_data20random_missing epochs 10 mape:52.56075549245972, rmse:5.256047521575037
    # cnn_bilstm_ae data all_train_data20random_missing epochs 20 mape:33.89199739813095, rmse:3.558607591605599
    # cnn_bilstm_ae data all_train_data20random_missing epochs 20 mape:32.82183410846747, rmse:3.2335844563841434 # 接着上面练的效果
    # cnn_bilstm_ae data all_train_data20random_missing epochs 60 mape:35.13418559300906, rmse:3.838446557630983
    # cnn_bilstm_ae_data_all_train_data20random_missing_epochs_15_mape = 32.76027585332283
    # cnn_bilstm_ae_data_all_train_data20random_missing_epochs_15_rmse = 3.422057700413118

    # cnn_bilstm_ae data all_train_data40random_missing epochs 5 mape:46.68132226431355, rmse:5.024397234986541
    # cnn_bilstm_ae data all_train_data60random_missing epochs 5 mape:62.59869249115234, rmse:6.701086022373199
    # cnn_bilstm_ae data all_train_data80random_missing epochs 5 mape:80.62183689220478, rmse:8.45387840207582
    # cnn_bilstm_ae data all_train_data20no_random_missing_road mape:41.8645175795609, rmse:4.474176159777803  epochs=1时效果最好，搞不懂
    # cnn_bilstm_ae data all_train_data20no_random_missing_road mape:66.36548494405442, rmse:7.256306568946944 epochs=5时
    # cnn_bilstm_ae data all_train_data20no_random_missing_road epochs 5 mape:74.81969503472197, rmse:7.912308174781946
    # cnn_bilstm_ae data all_train_data40no_random_missing_road epochs 5 mape:90.19086515517166, rmse:9.146217197629046
    # cnn_bilstm_ae data all_train_data60no_random_missing_road epochs 5 mape:96.03962657017266, rmse:9.869293497763307
    # cnn_bilstm_ae data all_train_data80no_random_missing_road epochs 5 mape:100.64908320109087, rmse:10.263565652065465
    # cnn_bilstm_ae data all_train_data20no_random_missing_time epochs 5 mape:99.8456227027119, rmse:10.271177296545703
    # cnn_bilstm_ae data all_train_data40no_random_missing_time epochs 5 mape:100.06365599496681, rmse:10.248421060175941
    # cnn_bilstm_ae data all_train_data60no_random_missing_time epochs 5 mape:100.9564949906909, rmse:10.293477587891715
    # cnn_bilstm_ae data all_train_data80no_random_missing_time epochs 5 mape:94.24154071418266, rmse:9.707489318228742

    # cnn_bilstm_ae data all_train_data80no_random_missing_time epochs 20 mape:100.6941742699254, rmse:10.33109827188591
    # cnn_bilstm_ae data all_train_data80no_random_missing_time epochs 30 mape:100.00793199423359, rmse:10.220162292685265
    # cnn_bilstm_ae data all_train_data60no_random_missing_time epochs 5 mape:99.62129940465161, rmse:10.192232331875804
    # cnn_bilstm_ae data all_train_data60no_random_missing_time epochs 20 mape:99.95314977876144, rmse:10.21753062560645
    # cnn_bilstm_ae data all_train_data40no_random_missing_time epochs 5 mape:99.86851379149331, rmse:10.241183921169176
    # cnn_bilstm_ae data all_train_data40no_random_missing_time epochs 10 mape:100.05137846971314, rmse:10.249888929832366
    # cnn_bilstm_ae data all_train_data20no_random_missing_time epochs 10 mape:99.9369104791374, rmse:10.282260595388736
    # cnn_bilstm_ae data all_train_data20no_random_missing_time epochs 5 mape:100.24871792178035, rmse:10.300565235523008
    # cnn_bilstm_ae data all_train_data20no_random_missing_time epochs 1 mape:99.98651610396016, rmse:10.221175312970077
    # cnn_bilstm_ae data all_train_data20no_random_missing_time epochs 20 mape:99.98208756260057, rmse:10.28597955538929
    # cnn_bilstm_ae data all_train_data20no_random_missing_time epochs 3 mape:99.8675258252135, rmse:10.261612717426532

if __name__ == '__main__':
    train_model( 100)
    # gen_data()
    # eval_model("all_train_data20random_missing")