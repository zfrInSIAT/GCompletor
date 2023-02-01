import sys

import numpy as np
import tensorflow as tf
from utils import get_deviation

sys.path.append(r'./baselines')
from baselines.metrics import mape, rmse
from baselines import TRMF, BN, BPMF, BTRMF, BGCP, BPTF, BTRTF, BTTF, LRTC_TNN, TRTF
from baselines.KNN import knn
from baselines.PPCA import PPCA


def t(speed_feature, train_speed_feature, mask,  missing_rate):
    '''
    换种求平均的方式看看
    '''
    train_speed_feature = np.reshape(train_speed_feature, [500,4,-1])
    tmp_mask = train_speed_feature != 0
    non_zero_nums = tf.math.count_nonzero(train_speed_feature, axis=-1, keepdims=True ,dtype='float64')
    speed_sum = tf.math.reduce_sum(train_speed_feature, axis=-1, keepdims=True)
    speed_mean = tf.math.divide(speed_sum,non_zero_nums)
    mean_value = np.mean(speed_mean[np.isfinite(speed_mean)])
    speed_mean = np.where(np.isfinite(speed_mean),speed_mean,mean_value)
    # print(speed_mean.shape)
    new_speed = tf.where(tmp_mask, train_speed_feature, speed_mean)
    new_speed = np.reshape(new_speed, [500, -1])
    # print(speed_feature.shape,new_speed.shape)
    # print(np.isnan(speed_feature[mask]).any())
    # print(np.isnan(new_speed[mask]).any())

    sorted_list = np.load(r'./data/sorted_var_500_road.npy')
    a = speed_feature[sorted_list[400:]]
    b = new_speed[sorted_list[400:]]
    m = mask[sorted_list[400:]]
    BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    c = speed_feature[sorted_list[:400]]
    d = new_speed[sorted_list[:400]]
    e = mask[sorted_list[:400]]
    BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

    print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

    mean_res2_mape2 = mape(speed_feature[mask], new_speed[mask])
    mean_res2_rmse2 = rmse(speed_feature[mask], new_speed[mask])
    ret_str = "mean_missing_rate{}mape={}\n".format(missing_rate, mean_res2_mape2)
    ret_str += "mean_missing_rate{}rmse={}".format(missing_rate, mean_res2_rmse2)
    print(ret_str)
# true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature=np.load('./feature.npz').values()
#
# speed_feature=np.squeeze(speed_feature,axis=-1)     # 把三维（288，4248，1）最后一个维度去掉
# train_speed_feature = np.squeeze(train_speed_feature,axis=-1)
#
# mask = true_mask ^ train_mask       # 两个掩码相异或，得到的才是在训练中没有，但我们又能得到的数据。
#
# mask = np.squeeze(mask,-1)
# train_mask=np.squeeze(train_mask,-1)
#
# assert len(speed_feature.shape) == 2 and len(train_speed_feature.shape)==2 and len(mask.shape)==2

# 在那些baselines的设置里是np.nan表示所处位置缺失，所以需要train_mask掩码来处理一下
def mean_exp(speed_feature, train_speed_feature,true_mask_mat, mask, missing_rate):
    # 有8条路，在14天的数据中都没有数据,所以可以注释掉了
    data = train_speed_feature.copy()
    # (4248,14*288)

    data_sum = np.sum(data, axis=-1)
    data_nums = np.count_nonzero(data, axis=-1)
    mean_data = np.divide(data_sum,data_nums)
    mean_value = np.mean(mean_data[np.isfinite(mean_data)])

    mean_data[np.isinf(mean_data)]=mean_value
    mean_data[np.isnan(mean_data)]=mean_value
    data = np.where(data != 0, data, np.expand_dims(mean_data, axis=-1))
    # big_var_road = np.load(r'./data/big_var_road.npy')
    # small_var_road = np.load(r'./data/small_var_road.npy')
    # a = speed_feature[big_var_road]
    # b = data[big_var_road]
    # m = mask[big_var_road]
    # BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    # BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    # c = speed_feature[small_var_road]
    # d = data[small_var_road]
    # e = mask[small_var_road]
    # BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    # BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])
    #
    # print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    # print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

    sorted_list = np.load(r'./data/sorted_var_500_road.npy')
    a = speed_feature[sorted_list[400:]]
    b = data[sorted_list[400:]]
    m = mask[sorted_list[400:]]
    BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    c = speed_feature[sorted_list[:400]]
    d = data[sorted_list[:400]]
    e = mask[sorted_list[:400]]
    BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

    print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

    mean_res2_mape2 = mape(speed_feature[mask], data[mask])
    mean_res2_rmse2 = rmse(speed_feature[mask], data[mask])
    # ret_str = "mean_missing_rate{}mape={}\n".format(missing_rate, mean_res2_mape2)
    # ret_str += "mean_missing_rate{}rmse={}".format(missing_rate, mean_res2_rmse2)
    ret_str = "mean_missing_rate{}mape={}\n".format(missing_rate, mean_res2_mape2)
    ret_str += "mean_missing_rate{}rmse={}".format(missing_rate, mean_res2_rmse2)
    print(ret_str)

    deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_mat, data)
    print("BTRMF_deviation_pret_data", deviation_pret_data)
    print("deviation_true_data", deviation_true_data)

    speed_feature_ten = np.reshape(speed_feature, [500, -1, 288])
    data_ten = np.reshape(data, [500,-1,288])
    mask_ten = np.reshape(mask, [500,-1,288])
    str2 = "分时段\n"
    for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
        t1 = mask_ten[..., item[0]:item[1]]
        t2 = speed_feature_ten[..., item[0]:item[1]][t1]
        t3 = data_ten[..., item[0]:item[1]][t1]
        if len(t2) == 0:
            mean_res_mape1 = 99999
            mean_res_rmse1 = 99999
        else:
            mean_res_mape1 = mape(t2, t3)
            mean_res_rmse1 = rmse(t2, t3)
        str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
        # str2 += "mape:{}\n".format(mean_res_mape1)
    print(str2)
    # 时间段在前，路段在后，从统计图上看，每个时间段都有值，所以不会出现mean一个空列表的情况

    # # 换个轴，再来一遍
    # data = np.transpose(train_speed_feature.copy())
    # data = np.where(data != 0, data, np.nan)
    # for i in range(data.shape[0]):
    #     meanVal = np.mean(data[i, ~np.isnan(data[i, :])])
    #     data[i, np.isnan(data[i, :])] = meanVal
    # data = np.transpose(data)
    # mean_res2_mape2 = mape(speed_feature[mask], data[mask])
    # mean_res2_rmse2 = rmse(speed_feature[mask], data[mask])
    # ret_str = "mean_missing_rate{}mape={}\n".format(missing_rate, mean_res2_mape2)
    # ret_str += "mean_missing_rate{}rmse={}".format(missing_rate, mean_res2_rmse2)
    # print(ret_str)
    # mean missing_rate 20 mape:58.23010255135347, rmse:4.5271115629201555
    # mean missing_rate 40 mape:58.16150836789827, rmse:4.529473292064819
    # mean missing_rate 60 mape:58.112858889068356, rmse:4.53208209662376
    # mean missing_rate 80 mape:58.174700838462954, rmse:4.532700804262092


def BTRMF_exp(speed_feature, train_speed_feature, true_mask_mat, mask,  missing_rate):
    '''

    :param speed_feature: 原始的数据矩阵
    :param train_speed_feature: 训练时用的数据矩阵
    :param true_all_mask_mat: 原始数据矩阵的掩码
    :param mask: 训练时没有，但是原始数据中又有的数据的掩码，用于计算补全精度
    :param missing_rate: 丢失率，用于输出提示输入数据的缺失程度
    :return:
    '''
    ret_str = ""
    for i in [7]:
        BTRMF_pre_speed = BTRMF.BTRMF(train_speed_feature, rank=i, time_lags=(1, 2), gibbs_iter=20, burn_iter=10)
        # big_var_road = np.load(r'./data/big_var_road.npy')
        # small_var_road = np.load(r'./data/small_var_road.npy')
        # a = speed_feature[big_var_road]
        # b = BTRMF_pre_speed[big_var_road]
        # m = mask[big_var_road]
        # BTRMF_res2_mape2_big_var = mape(a[m], b[m])
        # BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
        # c = speed_feature[small_var_road]
        # d = BTRMF_pre_speed[small_var_road]
        # e = mask[small_var_road]
        # BTRMF_res2_mape2_small_var = mape(c[e], d[e])
        # BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])
        #
        # print("大方差的道路: mape: ",BTRMF_res2_mape2_big_var,"rmse: ",BTRMF_res2_rmse2_big_var)
        # print("小方差的道路: mape: ",BTRMF_res2_mape2_small_var,"rmse: ",BTRMF_res2_rmse2_small_var)

        sorted_list = np.load(r'./data/sorted_var_500_road.npy')
        a = speed_feature[sorted_list[400:]]
        b = BTRMF_pre_speed[sorted_list[400:]]
        m = mask[sorted_list[400:]]
        BTRMF_res2_mape2_big_var = mape(a[m], b[m])
        BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
        c = speed_feature[sorted_list[:400]]
        d = BTRMF_pre_speed[sorted_list[:400]]
        e = mask[sorted_list[:400]]
        BTRMF_res2_mape2_small_var = mape(c[e], d[e])
        BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

        print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
        print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)


        BTRMF_res2_mape2 = mape(speed_feature[mask], BTRMF_pre_speed[mask])
        BTRMF_res2_rmse2 = rmse(speed_feature[mask], BTRMF_pre_speed[mask])
        # if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_mat, BTRMF_pre_speed)
        print("BTRMF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)
        ret_str += "BTRMF_missing_rate{},rank={},mape={},rmse={}\n".format(missing_rate, i, BTRMF_res2_mape2,BTRMF_res2_rmse2)
        ret_str += "BTRMF_missing_rate{}rmse={}\n".format(missing_rate, BTRMF_res2_rmse2)
        print(ret_str)

        speed_feature_ten = np.reshape(speed_feature, [500, -1, 288])
        data_ten = np.reshape(BTRMF_pre_speed, [500, -1, 288])
        mask_ten = np.reshape(mask, [500, -1, 288])
        str2 = "分时段\n"
        for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
            t1 = mask_ten[..., item[0]:item[1]]
            t2 = speed_feature_ten[..., item[0]:item[1]][t1]
            t3 = data_ten[..., item[0]:item[1]][t1]
            if len(t2) == 0:
                mean_res_mape1 = 99999
                mean_res_rmse1 = 99999
            else:
                mean_res_mape1 = mape(t2, t3)
                mean_res_rmse1 = rmse(t2, t3)
            str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
            # str2 += "mape:{}\n".format(mean_res_mape1)
        print(str2)
    # with open("true_data_and_pret_data","a+") as file:
    #     file.write("true_data="+str(speed_feature[mask].tolist())+"\n")
    #     file.write("BTRMF_pre_data="+str(BTRMF_pre_speed[mask].tolist())+"\n")
    # print("true_data=",speed_feature[mask])
    # print("BTRMF_pre_data=",BTRMF_pre_speed[mask])
    return ret_str


def BPMF_exp(speed_feature, train_speed_feature,true_mask_mat, mask, missing_rate):
    BPMF_pre_speed = BPMF.BPMF(train_speed_feature, rank=100, burn_iter=30, gibbs_iter=20)
    BPMF_res2_mape2 = mape(speed_feature[mask], BPMF_pre_speed[mask])
    BPMF_res2_rmse2 = rmse(speed_feature[mask], BPMF_pre_speed[mask])
    ret_str = "BPMF_missing_rate{}mape={}\n".format(missing_rate, BPMF_res2_mape2)
    ret_str += "BPMF_missing_rate{}rmse={}".format(missing_rate, BPMF_res2_rmse2)
    # print(ret_str)
    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_mat, BPMF_pre_speed)
        print("BPMF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)
    print(ret_str)
    return ret_str
    # BPMF mape:29.140189016558498, rmse:2.175620416365243
    # BPMF missing_rate 20 mape:29.001872626665374, rmse:2.080440829728151
    # BPMF missing_rate 40 mape:29.2140245316139, rmse:2.1023531311582255
    # BPMF missing_rate 60 mape:29.56153134397831, rmse:2.1297859118038245
    # BPMF missing_rate 80 mape:30.21426187196402, rmse:2.177547506144545


def BN_exp(speed_feature, train_speed_feature, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0., train_speed_feature, np.nan)
    pre_speed_feature = BN.BayesGauss(train_speed_feature_copy)
    BN_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    BN_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])
    res_str = ("BN missing_rate {} mape:{}, rmse:{}".format(missing_rate, BN_res2_mape2, BN_res2_rmse2))
    return res_str
    # BN mape:58.11579783055178, rmse:4.812629484157492


def TRMF_exp(speed_feature, train_speed_feature, true_mask_mat, mask, missing_rate):
    TRMF_pre_speed = TRMF.TRMF(train_speed_feature,rank=30, time_lags=(1, 2, 288),maxiter=50)
    TRMF_res2_mape2 = mape(speed_feature[mask], TRMF_pre_speed[mask])
    TRMF_res2_rmse2 = rmse(speed_feature[mask], TRMF_pre_speed[mask])
    ret_str = "TRMF_missing_rate{}mape={}\n".format(missing_rate, TRMF_res2_mape2)
    ret_str += "TRMF_missing_rate{}rmse={}".format(missing_rate, TRMF_res2_rmse2)
    print(ret_str)
    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_mat, TRMF_pre_speed)
        print("TRMF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)
    # print(ret_str)
    return ret_str
    # TRMF mape:30.534623798101602, rmse:2.2355243673299308
    # TRMF missing_rate 20 mape:28.787282747381067, rmse:2.0499284067913868
    # TRMF missing_rate 40 mape:28.884631618701643, rmse:2.064359073184114
    # TRMF missing_rate 60 mape:29.193423537076285, rmse:2.08866917505335
    # TRMF missing_rate 80 mape:29.771098434241676, rmse:2.1461994032787985


def PPCA_exp(speed_feature, train_speed_feature, true_mask_mat, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    pre_speed_feature = PPCA(train_speed_feature_copy, d=50)
    # res_str=(pre_speed_feature.shape)
    PPCA_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    PPCA_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])

    ret_str = "PPCA_missing_rate{}mape={}\n".format(missing_rate, PPCA_res2_mape2)
    ret_str += "PPCA_missing_rate{}rmse={}".format(missing_rate, PPCA_res2_rmse2)
    print(ret_str)
    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_mat, pre_speed_feature)
        print("PPCA_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)
    # print(res_str)
    return ret_str
    # 30.234063882822614 2.4240029917533783
    # PPCA missing_rate 20 mape:29.04058365342378, rmse:2.0811870306158813
    # PPCA missing_rate 40 mape:29.244637714873846, rmse:2.1020344518308955
    # PPCA missing_rate 60 mape:29.57371968447588, rmse:2.1303319638809577
    # PPCA missing_rate 80 mape:30.39575633456934, rmse:2.14256672741052


def KNN_exp(speed_feature, train_speed_feature,true_mask_mat, mask, missing_rate):
    # print(train_mask.shape, train_speed_feature.shape)
    # 按时间缺失时不可用
    # 去掉数据中全为0的列
    print("KNN experiment")
    flag=[]
    # all_zero_road =[]
    for i in range(speed_feature.shape[1]):
        # t=False
        # for j in range(speed_feature.shape[0]):
        #     if train_speed_feature[j,i]!=0:
        #         t=True
        #         break
        if np.count_nonzero(train_speed_feature[:,i])==0:
            flag.append(i)
            # all_zero_road.append(i)
    print("去除全为0的列成功")
    train_speed_feature[:,flag] = 0.1
    speed_feature[:,flag] = 0.1
    # true_mask_mat[:,flag] = False
    # mask2 = mask[:,flag]
    # print(np.sum(mask),np.sum(mask2))
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    print("开始运行knn算法")
    pre_speed_feature = knn(train_speed_feature_copy, n_neighbors=3)
    print("运行knn算法结束")
    # big_var_road = np.load(r'./data/big_var_road.npy')
    # small_var_road = np.load(r'./data/small_var_road.npy')
    # a = speed_feature[big_var_road]
    # b = pre_speed_feature[big_var_road]
    # m = mask2[big_var_road]
    # BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    # BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    # c = speed_feature[small_var_road]
    # d = pre_speed_feature[small_var_road]
    # e = mask2[small_var_road]
    # BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    # BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])
    #
    # print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    # print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

    sorted_list = np.load(r'./data/sorted_var_500_road.npy')
    a = speed_feature[sorted_list[400:]]
    b = pre_speed_feature[sorted_list[400:]]
    m = mask[sorted_list[400:]]
    BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    c = speed_feature[sorted_list[:400]]
    d = pre_speed_feature[sorted_list[:400]]
    e = mask[sorted_list[:400]]
    BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

    print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

    # print(pre_speed_feature[mask2].shape,speed_feature[mask2].shape)
    KNN_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    KNN_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])
    ret_str = "KNN_missing_rate{}mape={}\n".format(missing_rate, KNN_res2_mape2)
    ret_str += "KNN_missing_rate{}rmse={}".format(missing_rate, KNN_res2_rmse2)
    print(ret_str)
    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_mat, pre_speed_feature)
        print("KNN_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)

    # speed_feature_ten = np.reshape(speed_feature, [500, -1, 288])
    # data_ten = np.reshape(pre_speed_feature, [500, -1, 288])
    # mask_ten = np.reshape(mask, [500, -1, 288])
    # str2 = "分时段\n"
    # for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
    #     mean_res_mape1 = mape(speed_feature_ten[..., item[0]:item[1]][mask_ten[..., item[0]:item[1]]],
    #                           data_ten[..., item[0]:item[1]][mask_ten[..., item[0]:item[1]]])
    #     mean_res_rmse1 = rmse(speed_feature_ten[..., item[0]:item[1]][mask_ten[..., item[0]:item[1]]],
    #                           data_ten[..., item[0]:item[1]][mask_ten[..., item[0]:item[1]]])
    #     str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
    # print(str2)

    return ret_str

    # 两个轴换一下，变成（14*288,4248）
    # pre_speed_feature = np.transpose(knn(np.transpose(train_speed_feature_copy),n_neighbors=10))
    # KNN_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    # KNN_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])
    # res_str=("KNN missing_rate {} mape:{}, rmse:{}".format(missing_rate, KNN_res2_mape2, KNN_res2_rmse2))
    # mape:30.470718525938977	rmse:2.2707958351311275

def KNN_exp2(speed_feature, train_speed_feature,true_mask_mat, mask, missing_rate):
    # print(train_mask.shape, train_speed_feature.shape)
    # 按时间缺失时不可用
    # 去掉数据中全为0的列
    train_speed_feature_ten = np.reshape(train_speed_feature, [500, -1, 288])
    speed_feature_ten = np.reshape(speed_feature, [500, -1, 288])
    true_mask_ten = np.reshape(true_mask_mat, [500,-1,288])
    mask_ten = np.reshape(mask, [500,-1,288])
    # tmp_mask =[]
    pre_speed_feature_list = []
    speed_feature_list = []
    print("KNN experiment")
    for j in range(500):
        flag=[]
        # all_zero_road =[]
        for i in range(train_speed_feature_ten.shape[-1]):
            # t=False
            # for j in range(speed_feature.shape[0]):
            #     if train_speed_feature[j,i]!=0:
            #         t=True
            #         break
            if np.count_nonzero(train_speed_feature_ten[j,:,i])==0:
                flag.append(i)
                # all_zero_road.append(i)
        # print("去除全为0的列成功")
        # if np.sum(flag)==0:
        #     continue
        # train_speed_feature2 = train_speed_feature_ten[j][:, flag]
        train_speed_feature_ten[j,:, flag]=0.1
        mask_ten[j,:,flag]=False
        true_mask_ten[j,:,flag] = False
        # speed_feature_list.append(speed_feature_ten[j][:, flag])
        # true_mask_ten = true_mask_ten[j, :, flag]
        # tmp_mask.append(mask_ten[j][:, flag])
        # print(np.sum(mask),np.sum(mask2))
        train_speed_feature_copy = np.where(train_speed_feature_ten[j] != 0, train_speed_feature_ten[j], np.nan)
        # print("开始运行knn算法")
        pre_speed_feature_list.append(knn(train_speed_feature_copy, n_neighbors=3))
        # print("运行knn算法结束")
    # mask2 = np.concatenate(tmp_mask, axis=-1)
    pre_speed_feature = np.stack(pre_speed_feature_list, axis=0)
    # speed_feature3 = np.concatenate(speed_feature_list, axis=-1)
    KNN_res2_mape2 = mape(speed_feature_ten[mask_ten], pre_speed_feature[mask_ten])
    KNN_res2_rmse2 = rmse(speed_feature_ten[mask_ten], pre_speed_feature[mask_ten])
    ret_str = "KNN_missing_rate{}mape={}\n".format(missing_rate, KNN_res2_mape2)
    ret_str += "KNN_missing_rate{}rmse={}".format(missing_rate, KNN_res2_rmse2)
    print(ret_str)

    sorted_list = np.load(r'./data/sorted_var_500_road.npy')
    a = speed_feature_ten[sorted_list[400:]]
    b = pre_speed_feature[sorted_list[400:]]
    m = mask_ten[sorted_list[400:]]
    BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    c = speed_feature_ten[sorted_list[:400]]
    d = pre_speed_feature[sorted_list[:400]]
    e = mask_ten[sorted_list[:400]]
    BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

    print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

    speed_feature_ten = speed_feature_ten
    data_ten = pre_speed_feature
    mask_ten = mask_ten
    str2 = "分时段\n"
    for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
        t1 = mask_ten[..., item[0]:item[1]]
        t2 = speed_feature_ten[..., item[0]:item[1]][t1]
        t3 = data_ten[..., item[0]:item[1]][t1]
        if len(t2) == 0:
            mean_res_mape1 = 99999
            mean_res_rmse1 = 99999
        else:
            mean_res_mape1 = mape(t2, t3)
            mean_res_rmse1 = rmse(t2, t3)
        str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
    print(str2)
    # with open("true_data_and_pret_data","a+") as file:
    #     file.write("true_data="+str(speed_feature3[mask2].tolist())+"\n")
    #     file.write("BTRMF_pre_data="+str(pre_speed_feature[mask2].tolist())+"\n")
    # print("true_data=",speed_feature3[mask2])
    # print("KNN_pre_data=",pre_speed_feature[mask2])
    # if missing_rate == 20 or missing_rate=='20no_random_missing_road':
    deviation_true_data, deviation_pret_data = get_deviation(speed_feature_ten, true_mask_ten, pre_speed_feature)
    print("KNN_deviation_pret_data", deviation_pret_data)
    print("deviation_true_data", deviation_true_data)
    return ret_str
    # KNN_missing_rate20mape=41.83888206168827
    # KNN_missing_rate20rmse=2.6778689330175904

# 输入为张量的方法 BGCP ,BPTF, BTRTF, BTTF,LRTC_TNN,TRTF

def BGCP_exp(speed_feature, train_speed_feature, true_mask_ten, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    pre_speed_feature = BGCP.BGCP(train_speed_feature_copy, rank=50,burn_iter=50,gibbs_iter=30)
    # print(pre_speed_feature.shape)

    sorted_list = np.load(r'./data/sorted_var_500_road.npy')
    a = speed_feature[sorted_list[400:]]
    b = pre_speed_feature[sorted_list[400:]]
    m = mask[sorted_list[400:]]
    BTRMF_res2_mape2_big_var = mape(a[m], b[m])
    BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
    c = speed_feature[sorted_list[:400]]
    d = pre_speed_feature[sorted_list[:400]]
    e = mask[sorted_list[:400]]
    BTRMF_res2_mape2_small_var = mape(c[e], d[e])
    BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

    print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
    print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

    BGCP_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    BGCP_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])

    ret_str = "BGCP_missing_rate{}mape={}\n".format(missing_rate, BGCP_res2_mape2)
    ret_str += "BGCP_missing_rate{}rmse={}".format(missing_rate, BGCP_res2_rmse2)
    print(ret_str)

    speed_feature_ten = speed_feature
    data_ten = pre_speed_feature
    mask_ten = mask
    str2 = "分时段\n"
    for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
        t1 = mask_ten[..., item[0]:item[1]]
        t2 = speed_feature_ten[..., item[0]:item[1]][t1]
        t3 = data_ten[..., item[0]:item[1]][t1]
        if len(t2) == 0:
            mean_res_mape1 = 99999
            mean_res_rmse1 = 99999
        else:
            mean_res_mape1 = mape(t2, t3)
            mean_res_rmse1 = rmse(t2, t3)
        str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
    print(str2)

    # speed_feature_ten = np.reshape(speed_feature, [500, -1, 288])
    # data_ten = np.reshape(pre_speed_feature, [500, -1, 288])
    # mask_ten = np.reshape(mask, [500, -1, 288])
    # str2 = "分时段\n"
    # for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
    #     mean_res_mape1 = mape(speed_feature[..., item[0]:item[1]][mask[..., item[0]:item[1]]],
    #                           pre_speed_feature[..., item[0]:item[1]][mask[..., item[0]:item[1]]])
    #     mean_res_rmse1 = rmse(speed_feature[..., item[0]:item[1]][mask[..., item[0]:item[1]]],
    #                           pre_speed_feature[..., item[0]:item[1]][mask[..., item[0]:item[1]]])
    #     str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
    # print(str2)
    # with open("true_data_and_pret_data","a+") as file:
    #     file.write("true_data="+str(speed_feature[mask].tolist())+"\n")
    #     file.write("BTRMF_pre_data="+str(pre_speed_feature[mask].tolist())+"\n")
    # print("true_data=",speed_feature[mask])
    # print("BGCP_pre_data=",pre_speed_feature[mask])

    # if missing_rate == 20 or missing_rate=='20no_random_missing_road':
    deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_ten, pre_speed_feature)
    print("BGCP_deviation_pret_data", deviation_pret_data)
    print("deviation_true_data", deviation_true_data)
    # print(res_str)
    return ret_str


def BPTF_exp(speed_feature, train_speed_feature, true_mask_ten, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    pre_speed_feature = BPTF.BPTF(train_speed_feature_copy, rank=30, burn_iter=20, gibbs_iter=10)
    # print(pre_speed_feature.shape)
    BPTF_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    BPTF_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])

    ret_str = "BPTF_missing_rate{}mape={}\n".format(missing_rate, BPTF_res2_mape2)
    ret_str += "BPTF_missing_rate{}rmse={}".format(missing_rate, BPTF_res2_rmse2)
    print(ret_str)
    # print(res_str)
    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_ten, pre_speed_feature)
        print("BPTF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)
    # print(res_str)
    return ret_str


def BTRTF_exp(speed_feature, train_speed_feature,true_mask_ten, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    pre_speed_feature = BTRTF.BTRTF(train_speed_feature_copy, rank=10, time_lags=(1, 2, 288), burn_iter=20,
                                    gibbs_iter=10)
    # print(pre_speed_feature.shape)
    BTRTF_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    BTRTF_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])

    ret_str = "BTRTF_missing_rate{}mape={}\n".format(missing_rate, BTRTF_res2_mape2)
    ret_str += "BTRTF_missing_rate{}rmse={}".format(missing_rate, BTRTF_res2_rmse2)
    print(ret_str)
    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_ten, pre_speed_feature)
        print("BTRTF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)
    return ret_str


def BTTF_exp(speed_feature, train_speed_feature, true_mask_ten, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    pre_speed_feature = BTTF.BTTF(train_speed_feature_copy, rank=30, time_lags=(1, 2, 288), burn_iter=20,
                                  gibbs_iter=10)
    # print(pre_speed_feature.shape)
    BTTF_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    BTTF_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])

    ret_str = "BTTF_missing_rate{}mape={}\n".format(missing_rate, BTTF_res2_mape2)
    ret_str += "BTTF_missing_rate{}rmse={}".format(missing_rate, BTTF_res2_rmse2)
    print(ret_str)

    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_ten, pre_speed_feature)
        print("BTTF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)

    return ret_str


def LRTC_TNN_exp(speed_feature, train_speed_feature, true_mask_ten, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    pre_speed_feature = LRTC_TNN.LRTC_TNN(train_speed_feature_copy,maxiter=50)
    # print(pre_speed_feature.shape)
    LRTC_TNN_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    LRTC_TNN_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])

    ret_str = "LRTC_TNN_missing_rate{}mape={}\n".format(missing_rate, LRTC_TNN_res2_mape2)
    ret_str += "LRTC_TNN_missing_rate{}rmse={}".format(missing_rate, LRTC_TNN_res2_rmse2)
    print(ret_str)

    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_ten, pre_speed_feature)
        print("LRTC_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)

    return ret_str


def TRTF_exp(speed_feature, train_speed_feature, true_mask_ten, mask, missing_rate):
    train_speed_feature_copy = np.where(train_speed_feature != 0, train_speed_feature, np.nan)
    pre_speed_feature = TRTF.TRTF(train_speed_feature_copy, rank=30, time_lags=(1, 2, 288),maxiter=50)
    # print(pre_speed_feature.shape)
    TRTF_res2_mape2 = mape(speed_feature[mask], pre_speed_feature[mask])
    TRTF_res2_rmse2 = rmse(speed_feature[mask], pre_speed_feature[mask])

    ret_str = "TRTF_missing_rate{}mape={}\n".format(missing_rate, TRTF_res2_mape2)
    ret_str += "TRTF_missing_rate{}rmse={}".format(missing_rate, TRTF_res2_rmse2)
    print(ret_str)

    if missing_rate == 20 or missing_rate=='20no_random_missing_road':
        deviation_true_data, deviation_pret_data = get_deviation(speed_feature, true_mask_ten, pre_speed_feature)
        print("TRTF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)

    return ret_str


if __name__ == '__main__':
    true_all_data = np.load('./all_speed_data.npz')
    true_all_speed_feature, true_all_mask = true_all_data['arr_0'][10:], true_all_data['arr_1'][10:]
    # 保存的shape为（14，288，4248,1），需要squeeze一下
    true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature), np.squeeze(true_all_mask)

    # 以下shape均为（14，288，4248）, 随机缺失
    # data20miss = np.load("./data/all_train_data20random_missing.npz")
    # train_speed_data20miss, train_mask20miss = data20miss['arr_0'][10:], data20miss['arr_1'][10:]
    # data40miss = np.load("./data/all_train_data40random_missing.npz")
    # train_speed_data40miss, train_mask40miss = data40miss['arr_0'][10:], data40miss['arr_1'][10:]
    # data60miss = np.load("./data/all_train_data60random_missing.npz")
    # train_speed_data60miss, train_mask60miss = data60miss['arr_0'][10:], data60miss['arr_1'][10:]
    # data80miss = np.load("./data/all_train_data80random_missing.npz")
    # train_speed_data80miss, train_mask80miss = data80miss['arr_0'][10:], data80miss['arr_1'][10:]

    # 非随机缺失,按时间
    # data20miss = np.load("./data/all_train_data20no_random_missing_time.npz")
    # train_speed_data20miss, train_mask20miss = data20miss['arr_0'], data20miss['arr_1']
    # data40miss = np.load("./data/all_train_data40no_random_missing_time.npz")
    # train_speed_data40miss, train_mask40miss = data40miss['arr_0'], data40miss['arr_1']
    # data60miss = np.load("./data/all_train_data60no_random_missing_time.npz")
    # train_speed_data60miss, train_mask60miss = data60miss['arr_0'], data60miss['arr_1']
    # data80miss = np.load("./data/all_train_data80no_random_missing_time.npz")
    # train_speed_data80miss, train_mask80miss = data80miss['arr_0'], data80miss['arr_1']

    # 非随机缺失，按路段
    # data20miss = np.load("./data/all_train_data20no_random_missing_road.npz")
    # train_speed_data20miss, train_mask20miss = data20miss['arr_0'][10:], data20miss['arr_1'][10:]
    # data40miss = np.load("./data/all_train_data40no_random_missing_road.npz")
    # train_speed_data40miss, train_mask40miss = data40miss['arr_0'][10:], data40miss['arr_1'][10:]
    # data60miss = np.load("./data/all_train_data60no_random_missing_road.npz")
    # train_speed_data60miss, train_mask60miss = data60miss['arr_0'][10:], data60miss['arr_1'][10:]
    # data80miss = np.load("./data/all_train_data80no_random_missing_road.npz")
    # train_speed_data80miss, train_mask80miss = data80miss['arr_0'][10:], data80miss['arr_1'][10:]
    #
    # # 都得transpose一下，（m,n,f) n是天数，f是每天的离散时间间隔数
    true_all_speed_feature = true_all_speed_feature.transpose([2, 0, 1])
    true_all_mask = true_all_mask.transpose([2, 0, 1])
    # train_speed_data20miss = train_speed_data20miss.transpose([2, 0, 1])
    # train_speed_data40miss = train_speed_data40miss.transpose([2, 0, 1])
    # train_speed_data60miss = train_speed_data60miss.transpose([2, 0, 1])
    # train_speed_data80miss = train_speed_data80miss.transpose([2, 0, 1])
    # train_mask20miss = train_mask20miss.transpose([2, 0, 1])
    # train_mask40miss = train_mask40miss.transpose([2, 0, 1])
    # train_mask60miss = train_mask60miss.transpose([2, 0, 1])
    # train_mask80miss = train_mask80miss.transpose([2, 0, 1])
    # # 操作完之后的shape为（4248，14，288）
    #
    # mask20_ten = true_all_mask ^ train_mask20miss
    # mask40_ten = true_all_mask ^ train_mask40miss
    # mask60_ten = true_all_mask ^ train_mask60miss
    # mask80_ten = true_all_mask ^ train_mask80miss

    # 感觉这样直接转换会出问题，在其他代码中，是行为路段，列为时间序列
    # 需要先换轴，在reshape
    # 转化后的shape为（4248,14*288)
    # 注意前面已经transpose过了
    true_all_speed_feature_mat = true_all_speed_feature.reshape(4248, -1)
    true_all_mask_mat = true_all_mask.reshape(4248, -1)
    # train_speed_data20miss_mat = train_speed_data20miss.reshape(4248, -1)
    # train_mask20miss_mat = train_mask20miss.reshape(4248, -1)
    # train_speed_data40miss_mat = train_speed_data40miss.reshape(4248, -1)
    # train_mask40miss_mat = train_mask40miss.reshape(4248, -1)
    # train_speed_data60miss_mat = train_speed_data60miss.reshape(4248, -1)
    # train_mask60miss_mat = train_mask60miss.reshape(4248, -1)
    # train_speed_data80miss_mat = train_speed_data80miss.reshape(4248, -1)
    # train_mask80miss_mat = train_mask80miss.reshape(4248, -1)
    # #
    # mask20_mat = train_mask20miss_mat ^ true_all_mask_mat
    # mask40_mat = train_mask40miss_mat ^ true_all_mask_mat
    # mask60_mat = train_mask60miss_mat ^ true_all_mask_mat
    # mask80_mat = train_mask80miss_mat ^ true_all_mask_mat

    # summary = open('./summary.txt', 'a')
    selected500road_list_and_adj_mat = np.load("./data/selected500road_index_and_adj_mat.npz")
    roadsectId_list = selected500road_list_and_adj_mat['arr_0']
    roadsect_adj_mat = selected500road_list_and_adj_mat['arr_1']

    # t(true_all_speed_feature_mat, train_speed_data20miss_mat,true_all_mask_mat, mask20_mat, '20random_missing')
    # print("mean")
    # mean_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data20miss_mat[roadsectId_list],true_all_mask_mat[roadsectId_list], mask20_mat[roadsectId_list], '20no_random_missing_time')
    # mean_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data40miss_mat[roadsectId_list], mask40_mat[roadsectId_list], '40no_random_missing_time')
    # mean_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data60miss_mat[roadsectId_list], mask60_mat[roadsectId_list], '60no_random_missing_time')
    # mean_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data80miss_mat[roadsectId_list], mask80_mat[roadsectId_list], '80no_random_missing_time')
    #
    # print("t")
    # t(true_all_speed_feature_mat[roadsectId_list], train_speed_data20miss_mat[roadsectId_list],
    #          mask20_mat[roadsectId_list], '20no_random_missing_time')
    # t(true_all_speed_feature_mat[roadsectId_list], train_speed_data40miss_mat[roadsectId_list],
    #          mask40_mat[roadsectId_list], '40no_random_missing_time')
    # t(true_all_speed_feature_mat[roadsectId_list], train_speed_data60miss_mat[roadsectId_list],
    #          mask60_mat[roadsectId_list], '60no_random_missing_time')
    # t(true_all_speed_feature_mat[roadsectId_list], train_speed_data80miss_mat[roadsectId_list],
    #          mask80_mat[roadsectId_list], '80no_random_missing_time')
    # #
    # print("BTRMF")
    # a=BTRMF_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data20miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list], mask20_mat[roadsectId_list], '20no_random_missing_road')
    # b=BTRMF_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data40miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list], mask40_mat[roadsectId_list], '40no_random_missing_road')
    # c=BTRMF_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data60miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list], mask60_mat[roadsectId_list], '60no_random_missing_road')
    # d=BTRMF_exp(true_all_speed_feature_mat[roadsectId_list], train_speed_data80miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list], mask80_mat[roadsectId_list], '80no_random_missing_road')
    # summary.write(a + '\n')
    # summary.write(b + '\n')
    # summary.write(c + '\n')
    # summary.write(d + '\n')
    #
    # a=BPMF_exp(true_all_speed_feature_mat, train_speed_data20miss_mat, true_all_mask_mat, mask20_mat, '20no_random_missing_road')
    # b=BPMF_exp(true_all_speed_feature_mat, train_speed_data40miss_mat, true_all_mask_mat, mask40_mat, '40no_random_missing_road')
    # c=BPMF_exp(true_all_speed_feature_mat, train_speed_data60miss_mat, true_all_mask_mat, mask60_mat, '60no_random_missing_road')
    # d=BPMF_exp(true_all_speed_feature_mat, train_speed_data80miss_mat, true_all_mask_mat, mask80_mat, '80no_random_missing_road')
    # summary.write(a + '\n')
    # summary.write(b + '\n')
    # summary.write(c + '\n')
    # summary.write(d + '\n')

    # 因为无论矩阵是按路段在前还是时间在前排列，都有一些列全为空，即有一些路14天都没数据，而且有一些时间点，所有路段都没数据，因此bn方法不能用
    # a=BN_exp(np.transpose(true_all_speed_feature_mat), np.transpose(train_speed_data20miss_mat), np.transpose(mask20_mat), 20)
    # b=BN_exp(np.transpose(true_all_speed_feature_mat), np.transpose(train_speed_data40miss_mat), np.transpose(mask40_mat), 40)
    # c=BN_exp(np.transpose(true_all_speed_feature_mat), np.transpose(train_speed_data60miss_mat), np.transpose(mask60_mat), 60)
    # d=BN_exp(np.transpose(true_all_speed_feature_mat), np.transpose(train_speed_data80miss_mat), np.transpose(mask80_mat), 80)
    # summary.write(a + '\n')
    # summary.write(b + '\n')
    # summary.write(c + '\n')
    # summary.write(d + '\n')
    #
    # a=TRMF_exp(true_all_speed_feature_mat, train_speed_data20miss_mat, true_all_mask_mat, mask20_mat, '20no_random_missing_road')
    # b=TRMF_exp(true_all_speed_feature_mat, train_speed_data40miss_mat, true_all_mask_mat, mask40_mat, '40no_random_missing_road')
    # c=TRMF_exp(true_all_speed_feature_mat, train_speed_data60miss_mat, true_all_mask_mat, mask60_mat, '60no_random_missing_road')
    # d=TRMF_exp(true_all_speed_feature_mat, train_speed_data80miss_mat, true_all_mask_mat, mask80_mat, '80no_random_missing_road')
    # summary.write(a + '\n')
    # summary.write(b + '\n')
    # summary.write(c + '\n')
    # summary.write(d + '\n')
    #

    # 求不了分布
    # a=PPCA_exp(true_all_speed_feature_mat, train_speed_data20miss_mat, true_all_mask_mat, mask20_mat, 20)
    # b=PPCA_exp(true_all_speed_feature_mat, train_speed_data40miss_mat, true_all_mask_mat, mask40_mat, '40no_random_missing_time')
    # c=PPCA_exp(true_all_speed_feature_mat, train_speed_data60miss_mat, true_all_mask_mat, mask60_mat, '60no_random_missing_time')
    # d=PPCA_exp(true_all_speed_feature_mat, train_speed_data80miss_mat, true_all_mask_mat, mask80_mat, '80no_random_missing_time')
    # summary.write(a + '\n')
    # summary.write(b + '\n')
    # summary.write(c + '\n')
    # summary.write(d + '\n')

    # 同样的问题，当某一行全为0时，怎么判断和其他行的相似程度呢？所以不可用。
    # print("KNN 2")
    # a = KNN_exp2(true_all_speed_feature_mat[roadsectId_list], train_speed_data20miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list], mask20_mat[roadsectId_list], 20)
    # b = KNN_exp2(true_all_speed_feature_mat[roadsectId_list], train_speed_data40miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list],mask40_mat[roadsectId_list], 40)
    # c = KNN_exp2(true_all_speed_feature_mat[roadsectId_list], train_speed_data60miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list],mask60_mat[roadsectId_list], 60)
    # d = KNN_exp2(true_all_speed_feature_mat[roadsectId_list], train_speed_data80miss_mat[roadsectId_list], true_all_mask_mat[roadsectId_list],mask80_mat[roadsectId_list], 80)
    # summary.write(a + '\n')
    # summary.write(b + '\n')
    # summary.write(c + '\n')
    # summary.write(d + '\n')

    # 输入为tensor
    # print("BGCP")
    # a = BGCP_exp(true_all_speed_feature[roadsectId_list], train_speed_data20miss[roadsectId_list],true_all_mask[roadsectId_list], mask20_ten[roadsectId_list], '20no_random_missing_road')
    # # summary.write(a + '\n')
    # # BGCP missing_rate 20 mape:29.42907250486243, rmse:2.090028088372063 epochs=70
    # b = BGCP_exp(true_all_speed_feature[roadsectId_list], train_speed_data40miss[roadsectId_list],true_all_mask[roadsectId_list], mask40_ten[roadsectId_list], '40no_random_missing_road')
    # # summary.write(b + '\n')
    # # # BGCP missing_rate 40 mape:29.408380969782165, rmse:2.0966246419409735
    # c = BGCP_exp(true_all_speed_feature[roadsectId_list], train_speed_data60miss[roadsectId_list], true_all_mask[roadsectId_list],mask60_ten[roadsectId_list], '60no_random_missing_road')
    # # summary.write(c + '\n')
    # # # BGCP missing_rate 60 mape:29.402204433862405, rmse:2.0956020893533247
    # d = BGCP_exp(true_all_speed_feature[roadsectId_list], train_speed_data80miss[roadsectId_list],true_all_mask[roadsectId_list], mask80_ten[roadsectId_list], '80no_random_missing_road')
    # # # summary.write(d + '\n')
    # BGCP missing_rate 80 mape:29.63491950874635, rmse:2.108424623380917


    #
    # a = BPTF_exp(true_all_speed_feature, train_speed_data20miss,true_all_mask, mask20_ten, '20no_random_missing_road')
    # # summary.write(a + '\n')
    # # # BPTF missing_rate 20 mape:28.77276778699643, rmse:2.0543527503169763  epochs=900
    # #
    # b = BPTF_exp(true_all_speed_feature, train_speed_data40miss,true_all_mask, mask40_ten, '40no_random_missing_road')
    # # summary.write(b + '\n')
    # #
    # c = BPTF_exp(true_all_speed_feature, train_speed_data60miss,true_all_mask, mask60_ten, '60no_random_missing_road')
    # # summary.write(c + '\n')
    # #
    # d = BPTF_exp(true_all_speed_feature, train_speed_data80miss,true_all_mask, mask80_ten, '80no_random_missing_road')
    # summary.write(d + '\n')

    # BPTF missing_rate 20 mape:28.89461829599519, rmse:2.0632924607952434
    # BPTF missing_rate 40 mape:28.970857082786427, rmse:2.0760056102170883
    # BPTF missing_rate 60 mape:29.185009211132495, rmse:2.0917533336132648
    # BPTF missing_rate 80 mape:29.63541868386787, rmse:2.1263189041121073



    # a = BTRTF_exp(true_all_speed_feature, train_speed_data20miss,true_all_mask, mask20_ten, '20no_random_missing_time')
    # # summary.write(a + '\n')
    # #
    # b = BTRTF_exp(true_all_speed_feature, train_speed_data40miss,true_all_mask, mask40_ten, '40no_random_missing_time')
    # # summary.write(b + '\n')
    # #
    # c = BTRTF_exp(true_all_speed_feature, train_speed_data60miss,true_all_mask, mask60_ten, '60no_random_missing_time')
    # # summary.write(c + '\n')
    # #
    # d = BTRTF_exp(true_all_speed_feature, train_speed_data80miss,true_all_mask, mask80_ten, '80no_random_missing_time')
    # summary.write(d + '\n')
    # BTRTF missing_rate 20 mape:28.95062025410385, rmse:2.060752199893603
    # BTRTF missing_rate 40 mape:29.15287374462891, rmse:2.0868222771547433
    # BTRTF missing_rate 60 mape:29.167676708032666, rmse:2.084467983925318
    # BTRTF missing_rate 80 mape:29.73855198262061, rmse:2.1301770483945184



    # a = BTTF_exp(true_all_speed_feature, train_speed_data20miss,true_all_mask, mask20_ten, '20no_random_missing_road')
    # # summary.write(a + '\n')
    # #
    # b = BTTF_exp(true_all_speed_feature, train_speed_data40miss,true_all_mask, mask40_ten, '40no_random_missing_road')
    # # summary.write(b + '\n')
    # #
    # c = BTTF_exp(true_all_speed_feature, train_speed_data60miss,true_all_mask, mask60_ten, '60no_random_missing_road')
    # # summary.write(c + '\n')
    # #
    # d = BTTF_exp(true_all_speed_feature, train_speed_data80miss,true_all_mask, mask80_ten, '80no_random_missing_road')
    # summary.write(d + '\n')
    # BTTF missing_rate 20 mape:28.845599104715202, rmse:2.056110284653181
    # BTTF missing_rate 40 mape:28.870496956275144, rmse:2.064950287441245
    # BTTF missing_rate 60 mape:29.06040700662661, rmse:2.0780792188202604
    # BTTF missing_rate 80 mape:29.43179796609768, rmse:2.1024251280913795



    # a = LRTC_TNN_exp(true_all_speed_feature, train_speed_data20miss,true_all_mask, mask20_ten, '20no_random_missing_road')
    # # summary.write(a + '\n')
    # #
    # b = LRTC_TNN_exp(true_all_speed_feature, train_speed_data40miss,true_all_mask, mask40_ten, '40no_random_missing_road')
    # # summary.write(b + '\n')
    # #
    # c = LRTC_TNN_exp(true_all_speed_feature, train_speed_data60miss,true_all_mask, mask60_ten, '60no_random_missing_road')
    # # summary.write(c + '\n')
    # #
    # d = LRTC_TNN_exp(true_all_speed_feature, train_speed_data80miss,true_all_mask, mask80_ten, '80no_random_missing_road')
    # summary.write(d + '\n')
    # LRTC_TNN missing_rate 20 mape:29.662566143034425, rmse:2.1404106649463768
    # LRTC_TNN missing_rate 40 mape:29.825564434566882, rmse:2.1562707119756794
    # LRTC_TNN missing_rate 60 mape:30.024677247186244, rmse:2.182244976911179
    # LRTC_TNN missing_rate 80 mape:31.203890864895158, rmse:2.3706795864978667



    # a = TRTF_exp(true_all_speed_feature, train_speed_data20miss,true_all_mask, mask20_ten, '20no_random_missing_road')
    # # summary.write(a + '\n')
    # #
    # b = TRTF_exp(true_all_speed_feature, train_speed_data40miss,true_all_mask, mask40_ten, '40no_random_missing_road')
    # # summary.write(b + '\n')
    # #
    # c = TRTF_exp(true_all_speed_feature, train_speed_data60miss,true_all_mask, mask60_ten, '60no_random_missing_road')
    # # summary.write(c + '\n')
    # #
    # d = TRTF_exp(true_all_speed_feature, train_speed_data80miss,true_all_mask, mask80_ten, '80no_random_missing_road')
    # # summary.write(d + '\n')
    # BTTF missing_rate 20 mape:28.825261377239, rmse:2.0593460807036665
    # BTTF missing_rate 40 mape:28.867394636591726, rmse:2.0724974241006633
    # BTTF missing_rate 60 mape:29.065703752468686, rmse:2.0925430682456243
    # BTTF missing_rate 80 mape:29.521287728199013, rmse:2.1314333113429194

    # 处理工作日的数据。shape(10,288,4248)
    # 载入随机缺失的数据。
    # print("处理随机缺失，工作日数据")
    # weekday20miss_ten = np.load(r'./data/weekday/weekday20miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday40miss_ten = np.load(r'./data/weekday/weekday40miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday60miss_ten = np.load(r'./data/weekday/weekday60miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday80miss_ten = np.load(r'./data/weekday/weekday80miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekday20miss_ten = np.load(r'./data/weekday/mask_weekday20miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # # mask_weekday40miss_ten = np.load(r'./data/weekday/mask_weekday40miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # # mask_weekday60miss_ten = np.load(r'./data/weekday/mask_weekday60miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # # mask_weekday80miss_ten = np.load(r'./data/weekday/mask_weekday80miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday_all_data_ten = np.load(r'./data/weekday/weekday_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekday_all_data_ten = np.load(r'./data/weekday/mask_weekday_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    #
    # weekday20miss_mat = weekday20miss_ten.reshape([500, -1])
    # weekday40miss_mat = weekday40miss_ten.reshape([500, -1])
    # weekday60miss_mat = weekday60miss_ten.reshape([500, -1])
    # weekday80miss_mat = weekday80miss_ten.reshape([500, -1])
    # mask_weekday20miss_mat = mask_weekday20miss_ten.reshape([500, -1])
    # mask_weekday40miss_mat = mask_weekday40miss_ten.reshape([500, -1])
    # mask_weekday60miss_mat = mask_weekday60miss_ten.reshape([500, -1])
    # mask_weekday80miss_mat = mask_weekday80miss_ten.reshape([500, -1])
    # weekday_all_data_mat = weekday_all_data_ten.reshape([500, -1])
    # mask_weekday_all_data_mat = mask_weekday_all_data_ten.reshape([500, -1])
    #
    # mean_exp(weekday_all_data_mat, weekday20miss_mat, mask_weekday_all_data_mat,mask_weekday20miss_mat ^ mask_weekday_all_data_mat,
    #          '20no_random_missing_road')
    # mean_exp(weekday_all_data_mat, weekday40miss_mat, mask_weekday40miss_mat ^ mask_weekday_all_data_mat,
    #          '40no_random_missing_road')
    # mean_exp(weekday_all_data_mat, weekday60miss_mat, mask_weekday60miss_mat ^ mask_weekday_all_data_mat,
    #          '60no_random_missing_road')
    # mean_exp(weekday_all_data_mat, weekday80miss_mat, mask_weekday80miss_mat ^ mask_weekday_all_data_mat,
    #          '80no_random_missing_road')
    # BGCP_exp(weekday_all_data_ten, weekday20miss_ten, mask_weekday_all_data_ten, mask_weekday20miss_ten^mask_weekday_all_data_ten, '20no_random_missing_road')
    # BGCP_exp(weekday_all_data_ten, weekday40miss_ten, mask_weekday_all_data_ten, mask_weekday40miss_ten^mask_weekday_all_data_ten, '40no_random_missing_road')
    # BGCP_exp(weekday_all_data_ten, weekday60miss_ten, mask_weekday_all_data_ten, mask_weekday60miss_ten^mask_weekday_all_data_ten, '60no_random_missing_road')
    # BGCP_exp(weekday_all_data_ten, weekday80miss_ten, mask_weekday_all_data_ten, mask_weekday80miss_ten^mask_weekday_all_data_ten, '80no_random_missing_road')
    #
    # # mean_exp(weekday_all_data, weekday20miss, mask_weekday20miss^mask_weekday_all_data, '20no_random_missing_time')
    # # mean_exp(weekday_all_data, weekday40miss, mask_weekday40miss^mask_weekday_all_data, '40no_random_missing_time')
    # # mean_exp(weekday_all_data, weekday60miss, mask_weekday60miss^mask_weekday_all_data, '60no_random_missing_time')
    # # mean_exp(weekday_all_data, weekday80miss, mask_weekday80miss^mask_weekday_all_data, '80no_random_missing_time')
    # KNN_exp2(weekday_all_data_mat, weekday20miss_mat, mask_weekday_all_data_mat, mask_weekday20miss_mat^mask_weekday_all_data_mat,
    #               '20no_random_missing_road')
    # KNN_exp(weekday_all_data_mat, weekday40miss_mat, mask_weekday_all_data_mat, mask_weekday40miss_mat^mask_weekday_all_data_mat,
    #           '40no_random_missing_road')
    # KNN_exp(weekday_all_data_mat, weekday60miss_mat, mask_weekday_all_data_mat, mask_weekday60miss_mat^mask_weekday_all_data_mat,
    #           '60no_random_missing_road')
    # KNN_exp(weekday_all_data_mat, weekday80miss_mat, mask_weekday_all_data_mat, mask_weekday80miss_mat^mask_weekday_all_data_mat,
    #           '80no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday20miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday20miss_mat ^ mask_weekday_all_data_mat,
    #         '20no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday40miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday40miss_mat ^ mask_weekday_all_data_mat,
    #         '40no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday60miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday60miss_mat ^ mask_weekday_all_data_mat,
    #         '60no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday80miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday80miss_mat ^ mask_weekday_all_data_mat,
    #         '80no_random_missing_road')


    # 处理周末的数据
    # print("随机缺失，处理周末数据")
    # weekend20miss_ten = np.load(r'./data/weekend/weekend20miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekend40miss_ten = np.load(r'./data/weekend/weekend40miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekend60miss_ten = np.load(r'./data/weekend/weekend60miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekend80miss_ten = np.load(r'./data/weekend/weekend80miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekend20miss_ten = np.load(r'./data/weekend/mask_weekend20miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekend40miss_ten = np.load(r'./data/weekend/mask_weekend40miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekend60miss_ten = np.load(r'./data/weekend/mask_weekend60miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekend80miss_ten = np.load(r'./data/weekend/mask_weekend80miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekend_all_data_ten = np.load(r'./data/weekend/weekend_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekend_all_data_ten = np.load(r'./data/weekend/mask_weekend_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    #
    # weekend20miss_mat = weekend20miss_ten.reshape([500, -1])
    # weekend40miss_mat = weekend40miss_ten.reshape([500, -1])
    # weekend60miss_mat = weekend60miss_ten.reshape([500, -1])
    # weekend80miss_mat = weekend80miss_ten.reshape([500, -1])
    # mask_weekend20miss_mat = mask_weekend20miss_ten.reshape([500, -1])
    # mask_weekend40miss_mat = mask_weekend40miss_ten.reshape([500, -1])
    # mask_weekend60miss_mat = mask_weekend60miss_ten.reshape([500, -1])
    # mask_weekend80miss_mat = mask_weekend80miss_ten.reshape([500, -1])
    # weekend_all_data_mat = weekend_all_data_ten.reshape([500, -1])
    # mask_weekend_all_data_mat = mask_weekend_all_data_ten.reshape([500, -1])
    # BGCP_exp(weekend_all_data_ten, weekend20miss_ten, mask_weekend_all_data_ten, mask_weekend20miss_ten ^ mask_weekend_all_data_ten,
    #          '20no_random_missing_road')
    # BGCP_exp(weekend_all_data_ten, weekend40miss_ten, mask_weekend_all_data_ten, mask_weekend40miss_ten ^ mask_weekend_all_data_ten,
    #          '40no_random_missing_road')
    # BGCP_exp(weekend_all_data_ten, weekend60miss_ten, mask_weekend_all_data_ten, mask_weekend60miss_ten ^ mask_weekend_all_data_ten,
    #          '60no_random_missing_road')
    # BGCP_exp(weekend_all_data_ten, weekend80miss_ten, mask_weekend_all_data_ten, mask_weekend80miss_ten ^ mask_weekend_all_data_ten,
    #          '80no_random_missing_road')
    # mean_exp(weekend_all_data_mat, weekend20miss_mat, mask_weekend_all_data_mat ,mask_weekend20miss_mat^mask_weekend_all_data_mat , '20no_random_missing_time')
    # mean_exp(weekend_all_data_mat, weekend40miss_mat, mask_weekend40miss_mat^mask_weekend_all_data_mat, '40no_random_missing_time')
    # mean_exp(weekend_all_data_mat, weekend60miss_mat, mask_weekend60miss_mat^mask_weekend_all_data_mat, '60no_random_missing_time')
    # mean_exp(weekend_all_data_mat, weekend80miss_mat, mask_weekend80miss_mat^mask_weekend_all_data_mat, '80no_random_missing_time')
    # KNN_exp2(weekend_all_data_mat, weekend20miss_mat, mask_weekend_all_data_mat, mask_weekend20miss_mat^mask_weekend_all_data_mat,
    #           '20no_random_missing_road')
    # KNN_exp(weekend_all_data_mat, weekend40miss_mat, mask_weekend_all_data_mat, mask_weekend40miss_mat^mask_weekend_all_data_mat,
    #           '40no_random_missing_road')
    # KNN_exp(weekend_all_data_mat, weekend60miss_mat, mask_weekend_all_data_mat, mask_weekend60miss_mat^mask_weekend_all_data_mat,
    #           '60no_random_missing_road')
    # KNN_exp(weekend_all_data_mat, weekend80miss_mat, mask_weekend_all_data_mat, mask_weekend80miss_mat^mask_weekend_all_data_mat,
    #           '80no_random_missing_road')
    # BTRMF_exp(weekend_all_data_mat, weekend20miss_mat,mask_weekend_all_data_mat, mask_weekend20miss_mat ^ mask_weekend_all_data_mat,
    #         '20no_random_missing_road')
    # BTRMF_exp(weekend_all_data_mat, weekend40miss_mat,  mask_weekend40miss_mat ^ mask_weekend_all_data_mat,
    #         '40no_random_missing_road')
    # BTRMF_exp(weekend_all_data_mat, weekend60miss_mat,  mask_weekend60miss_mat ^ mask_weekend_all_data_mat,
    #         '60no_random_missing_road')
    # BTRMF_exp(weekend_all_data_mat, weekend80miss_mat,  mask_weekend80miss_mat ^ mask_weekend_all_data_mat,
    #         '80no_random_missing_road')

    # 处理非随机缺失的数据
    # print("处理非随机缺失的数据，工作日")
    # weekday20miss_ten = np.load(r'./data/weekday/weekday20no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday40miss_ten = np.load(r'./data/weekday/weekday40no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday60miss_ten = np.load(r'./data/weekday/weekday60no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday80miss_ten = np.load(r'./data/weekday/weekday80no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekday20miss_ten = np.load(r'./data/weekday/mask_weekday20no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekday40miss_ten = np.load(r'./data/weekday/mask_weekday40no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekday60miss_ten = np.load(r'./data/weekday/mask_weekday60no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekday80miss_ten = np.load(r'./data/weekday/mask_weekday80no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday_all_data_ten = np.load(r'./data/weekday/weekday_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    # mask_weekday_all_data_ten = np.load(r'./data/weekday/mask_weekday_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    # weekday20miss_mat = weekday20miss_ten.reshape([500, -1])
    # weekday40miss_mat = weekday40miss_ten.reshape([500, -1])
    # weekday60miss_mat = weekday60miss_ten.reshape([500, -1])
    # weekday80miss_mat = weekday80miss_ten.reshape([500, -1])
    # mask_weekday20miss_mat = mask_weekday20miss_ten.reshape([500, -1])
    # mask_weekday40miss_mat = mask_weekday40miss_ten.reshape([500, -1])
    # mask_weekday60miss_mat = mask_weekday60miss_ten.reshape([500, -1])
    # mask_weekday80miss_mat = mask_weekday80miss_ten.reshape([500, -1])
    # weekday_all_data_mat = weekday_all_data_ten.reshape([500, -1])
    # mask_weekday_all_data_mat = mask_weekday_all_data_ten.reshape([500, -1])

    # BGCP_exp(weekday_all_data_ten, weekday20miss_ten, mask_weekday_all_data_ten, mask_weekday20miss_ten ^ mask_weekday_all_data_ten,
    #          '20no_random_missing_road')
    # BGCP_exp(weekday_all_data_ten, weekday40miss_ten, mask_weekday_all_data_ten, mask_weekday40miss_ten ^ mask_weekday_all_data_ten,
    #          '40no_random_missing_road')
    # BGCP_exp(weekday_all_data_ten, weekday60miss_ten, mask_weekday_all_data_ten, mask_weekday60miss_ten ^ mask_weekday_all_data_ten,
    #          '60no_random_missing_road')
    # BGCP_exp(weekday_all_data_ten, weekday80miss_ten, mask_weekday_all_data_ten, mask_weekday80miss_ten ^ mask_weekday_all_data_ten,
    #          '80no_random_missing_road')
    # KNN_exp2(weekday_all_data_mat, weekday20miss_mat, mask_weekday_all_data_mat, mask_weekday20miss_mat ^ mask_weekday_all_data_mat,
    #           '20no_random_missing_road')

    # KNN_exp(weekday_all_data_mat, weekday40miss_mat, mask_weekday_all_data_mat, mask_weekday40miss_mat ^ mask_weekday_all_data_mat,
    #           '40no_random_missing_road')
    # KNN_exp(weekday_all_data_mat, weekday60miss_mat, mask_weekday_all_data_mat, mask_weekday60miss_mat ^ mask_weekday_all_data_mat,
    #           '60no_random_missing_road')
    # KNN_exp(weekday_all_data_mat, weekday80miss_mat, mask_weekday_all_data_mat, mask_weekday80miss_mat ^ mask_weekday_all_data_mat,
    #           '80no_random_missing_road')
    # mean_exp(weekday_all_data_mat, weekday20miss_mat,mask_weekday_all_data_mat,mask_weekday20miss_mat ^ mask_weekday_all_data_mat,
    #         '20no_random_missing_road')
    # mean_exp(weekday_all_data_mat, weekday40miss_mat, mask_weekday40miss_mat ^ mask_weekday_all_data_mat,
    #         '40no_random_missing_road')
    # mean_exp(weekday_all_data_mat, weekday60miss_mat, mask_weekday60miss_mat ^ mask_weekday_all_data_mat,
    #         '60no_random_missing_road')
    # mean_exp(weekday_all_data_mat, weekday80miss_mat, mask_weekday80miss_mat ^ mask_weekday_all_data_mat,
    #         '80no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday20miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday20miss_mat ^ mask_weekday_all_data_mat,
    #         '20no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday40miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday40miss_mat ^ mask_weekday_all_data_mat,
    #         '40no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday60miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday60miss_mat ^ mask_weekday_all_data_mat,
    #         '60no_random_missing_road')
    # BTRMF_exp(weekday_all_data_mat, weekday80miss_mat, mask_weekday_all_data_mat,
    #         mask_weekday80miss_mat ^ mask_weekday_all_data_mat,
    #         '80no_random_missing_road')

    print("处理非随机缺失，周末数据")
    weekend20miss_ten = np.load(r'./data/weekend/weekend20no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    weekend40miss_ten = np.load(r'./data/weekend/weekend40no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    weekend60miss_ten = np.load(r'./data/weekend/weekend60no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    weekend80miss_ten = np.load(r'./data/weekend/weekend80no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    mask_weekend20miss_ten = np.load(r'./data/weekend/mask_weekend20no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    mask_weekend40miss_ten = np.load(r'./data/weekend/mask_weekend40no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    mask_weekend60miss_ten = np.load(r'./data/weekend/mask_weekend60no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    mask_weekend80miss_ten = np.load(r'./data/weekend/mask_weekend80no_random_miss.npy').transpose([2, 0, 1])[roadsectId_list]
    weekend_all_data_ten = np.load(r'./data/weekend/weekend_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    mask_weekend_all_data_ten = np.load(r'./data/weekend/mask_weekend_all_data.npy').transpose([2, 0, 1])[roadsectId_list]
    weekend20miss_mat = weekend20miss_ten.reshape([500, -1])
    weekend40miss_mat = weekend40miss_ten.reshape([500, -1])
    weekend60miss_mat = weekend60miss_ten.reshape([500, -1])
    weekend80miss_mat = weekend80miss_ten.reshape([500, -1])
    mask_weekend20miss_mat = mask_weekend20miss_ten.reshape([500, -1])
    mask_weekend40miss_mat = mask_weekend40miss_ten.reshape([500, -1])
    mask_weekend60miss_mat = mask_weekend60miss_ten.reshape([500, -1])
    mask_weekend80miss_mat = mask_weekend80miss_ten.reshape([500, -1])
    weekend_all_data_mat = weekend_all_data_ten.reshape([500, -1])
    mask_weekend_all_data_mat = mask_weekend_all_data_ten.reshape([500, -1])

    BGCP_exp(weekend_all_data_ten, weekend20miss_ten, mask_weekend_all_data_ten, mask_weekend20miss_ten ^ mask_weekend_all_data_ten,
             '20no_random_missing_road')
    # BGCP_exp(weekend_all_data_ten, weekend40miss_ten, mask_weekend_all_data_ten, mask_weekend40miss_ten ^ mask_weekend_all_data_ten,
    #          '40no_random_missing_road')
    # BGCP_exp(weekend_all_data_ten, weekend60miss_ten, mask_weekend_all_data_ten, mask_weekend60miss_ten ^ mask_weekend_all_data_ten,
    #          '60no_random_missing_road')
    # BGCP_exp(weekend_all_data_ten, weekend80miss_ten, mask_weekend_all_data_ten, mask_weekend80miss_ten ^ mask_weekend_all_data_ten,
    #          '80no_random_missing_road')
    KNN_exp2(weekend_all_data_mat, weekend20miss_mat, mask_weekend_all_data_mat, mask_weekend20miss_mat ^ mask_weekend_all_data_mat,
              '20no_random_missing_road')
    # KNN_exp(weekend_all_data_mat, weekend40miss_mat, mask_weekend_all_data_mat, mask_weekend40miss_mat ^ mask_weekend_all_data_mat,
    #           '40no_random_missing_road')
    # KNN_exp(weekend_all_data_mat, weekend60miss_mat, mask_weekend_all_data_mat, mask_weekend60miss_mat ^ mask_weekend_all_data_mat,
    #           '60no_random_missing_road')
    # KNN_exp(weekend_all_data_mat, weekend80miss_mat, mask_weekend_all_data_mat, mask_weekend80miss_mat ^ mask_weekend_all_data_mat,
    #           '80no_random_missing_road')

    mean_exp(weekend_all_data_mat, weekend20miss_mat, mask_weekend_all_data_mat,
            mask_weekend20miss_mat ^ mask_weekend_all_data_mat,"20")
    # mean_exp(weekend_all_data_mat, weekend40miss_mat, mask_weekend_all_data_mat,
    #         mask_weekend40miss_mat ^ mask_weekend_all_data_mat)
    # mean_exp(weekend_all_data_mat, weekend60miss_mat, mask_weekend_all_data_mat,
    #         mask_weekend60miss_mat ^ mask_weekend_all_data_mat)
    # mean_exp(weekend_all_data_mat, weekend80miss_mat, mask_weekend_all_data_mat,
    #         mask_weekend80miss_mat ^ mask_weekend_all_data_mat)

    BTRMF_exp(weekend_all_data_mat, weekend20miss_mat, mask_weekend_all_data_mat,
             mask_weekend20miss_mat ^ mask_weekend_all_data_mat,
             '20no_random_missing_road')
    # BTRMF_exp(weekend_all_data_mat, weekend40miss_mat, mask_weekend_all_data_mat,
    #          mask_weekend40miss_mat ^ mask_weekend_all_data_mat,
    #          '40no_random_missing_road')
    # BTRMF_exp(weekend_all_data_mat, weekend60miss_mat, mask_weekend_all_data_mat,
    #          mask_weekend60miss_mat ^ mask_weekend_all_data_mat,
    #          '60no_random_missing_road')
    # BTRMF_exp(weekend_all_data_mat, weekend80miss_mat, mask_weekend_all_data_mat,
    #          mask_weekend80miss_mat ^ mask_weekend_all_data_mat,
    #          '80no_random_missing_road')


