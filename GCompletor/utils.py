from __future__ import print_function

import os
from collections import defaultdict,deque

# import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tensorflow.keras.utils import to_categorical


def gen_edge_feature():
    speed_data = np.load('./all_speed_data.npz')
    # shape (14,288,4248,1)
    all_speed_data = speed_data['arr_0']
    all_speed_mask = speed_data['arr_1']
    # squeeze之后，再换一下轴
    all_speed_data = np.transpose(np.squeeze(all_speed_data), [2, 0, 1])  # shape(4248,14,288)
    all_speed_data_shape = all_speed_data.shape
    tmp_tensor = np.zeros(all_speed_data_shape)
    for i in tqdm.trange(all_speed_data_shape[0]):
        for j in range(all_speed_data_shape[1]):
            for k in range(1, all_speed_data_shape[2]):
                if all_speed_data[i, j, k - 1] == 0 or all_speed_data[i, j, k] == 0:
                    tmp_tensor[i, j, k] = i
                elif all_speed_data[i, j, k - 1] <= all_speed_data[i, j, k]:
                    tmp_tensor[i, j, k] = 1
                elif all_speed_data[i, j, k - 1] > all_speed_data[i, j, k]:
                    tmp_tensor[i, j, k] = -1
    # 边的特征构造出来，得是4248*4248的矩阵，用密集矩阵存储太耗内存，用SparseTensor好些
    _, adj_mat = get_roadId_list_and_adj_mat()
    indices = []
    values = [[] for _ in range(14)]
    dense_shape = adj_mat.shape
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[i])):
            if adj_mat[i][j] == 1:
                indices.append([i, j])
                for k in range(14):
                    values[k].append(sum(tmp_tensor[i, k] == tmp_tensor[j, k]) / 288)
    adj = [indices, values, dense_shape]
    np.save('edge_feature_sparse_tensor', adj)


def save_sparse_adj():
    _, adj_mat = get_roadId_list_and_adj_mat()

    # 把邻接矩阵变成spaesetensor
    indices = []
    values = []
    dense_shape = adj_mat.shape
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[i])):
            if adj_mat[i][j] == 1:
                indices.append([i, j])
                values.append(adj_mat[i][j])
    adj = [indices, values, dense_shape]
    np.save('sparse_adj', adj)


def get_deviation(true_data, mask, pret_data):
    '''

    :param true_data:
    :param mask: 这个掩码是原数据中有值为True，缺失为False
    :param pret_data:
    :return:
    '''
    true_data_mean = np.mean(true_data[mask])
    true_data_num = np.sum(mask)
    true_data_distribution_ret = []
    metrics = [-9.75, -9.25, -8.75, -8.25, -7.75, -7.25, -6.75, -6.25, -5.75, -5.25, -4.75, -4.25, -3.75, -3.25, -2.75,
               -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25,
               5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75]
    # true_data_distribution_ret.append(np.sum(true_data[mask]-true_data_mean<-10)/true_data_num)
    for i in range(len(metrics) - 1):
        t = (np.sum(true_data[mask] - true_data_mean < metrics[i + 1]) - np.sum(
            true_data[mask] - true_data_mean < metrics[i])) / true_data_num
        true_data_distribution_ret.append(t)
    # true_data_distribution_ret.append(np.sum(true_data[mask]-true_data_mean>10)/true_data_num)

    # 这里到底需不需要对mask进行求反？应该是不需要，因为就是为了判断真实的和预测的值得分布有何不同，才可以做对比
    # pret_mask = ~mask
    pret_mask = mask
    pret_data_mean = np.mean(pret_data[pret_mask])
    pret_data_num = np.sum(pret_mask)
    pret_data_distribution_ret = []
    # pret_data_distribution_ret.append(np.sum(pret_data[pret_mask]<-10)/pret_data_num)
    for i in range(len(metrics) - 1):
        t = (np.sum(pret_data[pret_mask] - pret_data_mean < metrics[i + 1]) - np.sum(
            pret_data[pret_mask] - pret_data_mean < metrics[i])) / pret_data_num
        pret_data_distribution_ret.append(t)
    # pret_data_distribution_ret.append(np.sum(pret_data[pret_mask]>10)/pret_data_num)
    return true_data_distribution_ret, pret_data_distribution_ret


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def get_roadId_list_and_adj_mat():
    arr = np.load('./data/roadsectId_list_and_adj_mat.npz')  # 文件中存储已经整理好的道路id的列表，和道路的邻接矩阵
    roadId_list, adj_mat = arr['arr_0'], arr['arr_1']
    return roadId_list, adj_mat


def read_speed_data4DSAE():
    roadId_list, _ = get_roadId_list_and_adj_mat()  # 道路id能保证不重复
    # print(len(np.unique(roadId_list)))
    speed_data_path = r'./data/data_category_by_date/2018-04-07'
    dic = defaultdict(list)
    speed_feature = np.zeros((288, 4248))
    # count_not_in_roadId_list=0
    with open(speed_data_path, 'r') as f:
        for line in f.readlines():
            if line is not None:
                line = line.split(' ')
                road_id, period, speed_val = int(line[0]), int(line[1]), float(line[2])
                dic[period].append([road_id, speed_val])
    for idx, item in dic.items():
        for i in item:

            pos = np.where(roadId_list == i[0])
            if len(pos[0]) != 1:
                continue
            else:
                speed_feature[idx - 1][pos[0][0]] = i[1]
    np.save('speed_feature4DSAE', speed_feature)
    return speed_feature


def read_speed_data4predict():
    roadId_list, _ = get_roadId_list_and_adj_mat()  # 道路id能保证不重复
    # print(len(np.unique(roadId_list)))
    speed_data_path = r'./data/data_category_by_date/2018-04-07'
    dic = defaultdict(list)
    speed_feature = np.zeros((288, 4248, 4))
    # count_not_in_roadId_list=0
    with open(speed_data_path, 'r') as f:
        for line in f.readlines():
            if line is not None:
                line = line.split(' ')
                road_id, period, speed_val = int(line[0]), int(line[1]), float(line[2])
                dic[period].append([road_id, speed_val])
    for idx, item in dic.items():
        for i in item:

            pos = np.where(roadId_list == i[0])
            if len(pos[0]) != 1:
                continue
                # print('error in roadId_list')
                # print(i[0])

                # 用于统计有多少条数据的道路id是不在道路列表中的，统计结果显示2018-04-07有1778条数据不合规
                # count_not_in_roadId_list+=1
            else:
                speed_feature[idx - 1][pos[0][0]][0] = i[1]
                speed_feature[idx - 1][pos[0][0]][1] = idx
                speed_feature[idx - 1][pos[0][0]][2] = 1
                speed_feature[idx - 1][pos[0][0]][3] = 1
    # mask = speed_feature != 0  # 速度特征的掩码，有数据的地方为True，没数据的地方为False
    np.save('road_speed_feature', speed_feature)
    return speed_feature


def read_all_speed_data():
    roadId_list, _ = get_roadId_list_and_adj_mat()  # 道路id能保证不重复
    # print(len(np.unique(roadId_list)))
    speed_data_path = r'./data/data_category_by_date'

    speed_feature = np.zeros((14, 288, 4248, 1))
    # count_not_in_roadId_list=0
    for t in range(14):
        file_name = '2018-04-' + str(t + 6).zfill(2)
        dic = defaultdict(list)
        with open(os.path.join(speed_data_path, file_name), 'r') as f:
            for line in f.readlines():
                if line is not None:
                    line = line.split(' ')
                    road_id, period, speed_val = int(line[0]), int(line[1]), float(line[2])
                    dic[period].append([road_id, speed_val])
        for idx, item in dic.items():
            for i in item:

                pos = np.where(roadId_list == i[0])
                if len(pos[0]) != 1:
                    continue
                    # print('error in roadId_list')
                    # print(i[0])

                    # 用于统计有多少条数据的道路id是不在道路列表中的，统计结果显示2018-04-07有1778条数据不合规
                    # count_not_in_roadId_list+=1
                else:
                    speed_feature[t][idx - 1][pos[0][0]][0] = i[1]
    mask = speed_feature != 0  # 速度特征的掩码，有数据的地方为True，没数据的地方为False
    np.savez('all_speed_data', speed_feature, mask)
    # 返回一个三维张量，（14，288，4248,1），14天内的速度特征和对应的掩码，这是能得到的尽量完整的数据。
    return speed_feature, mask


def all_speed_train_data():
    data = np.load(r'./all_speed_data.npz')
    speed_feature, mask = data['arr_0'], data['arr_1']
    speed_feature, mask = np.squeeze(speed_feature), np.squeeze(mask)  # 去掉最后一个维度,shape（14，288，4248）
    # 在数据本身就缺失的状况下（缺失程度约为20%），再分别随机丢掉20%，40%，60%，80%
    # rand = np.random.random()
    train_mask20miss = np.where(mask, np.random.rand(*mask.shape) < 0.8, mask)
    train_mask40miss = np.where(mask, np.random.rand(*mask.shape) < 0.6, mask)
    train_mask60miss = np.where(mask, np.random.rand(*mask.shape) < 0.4, mask)
    train_mask80miss = np.where(mask, np.random.rand(*mask.shape) < 0.2, mask)

    train_speed_data20miss = np.where(train_mask20miss, speed_feature, 0.)
    train_speed_data40miss = np.where(train_mask40miss, speed_feature, 0.)
    train_speed_data60miss = np.where(train_mask60miss, speed_feature, 0.)
    train_speed_data80miss = np.where(train_mask80miss, speed_feature, 0.)

    # 这里存储的是14天内所有的数据数据和它的掩码，都是再能得到的数据的基础上，再丢失20%，40%,60%,80%的数据，shape为（14，288，4248）
    np.savez(r'./data/all_train_data20random_missing', train_speed_data20miss, train_mask20miss)
    np.savez(r'./data/all_train_data40random_missing', train_speed_data40miss, train_mask40miss)
    np.savez(r'./data/all_train_data60random_missing', train_speed_data60miss, train_mask60miss)
    np.savez(r'./data/all_train_data80random_missing', train_speed_data80miss, train_mask80miss)

    # 注意，相比于随机缺失，非随机缺失，会导致最后的缺失率稍微大于设定的缺失率。
    # 随机缺失，是在原本缺失的基础上再丢掉20%，40%，60%，80%，原本14天的数据的缺失率为58.8%，最后缺失的结果为58%*[20%,40%,60%,80%]
    # 而非随机缺失，缺失率会比58%*[20%,40%,60%,80%]大一些。

    # 某一天的某个时刻，所有的路段的数据都发生缺失，非随机缺失的一种
    rand = np.random.random(speed_feature.shape[:-1])
    train_speed_data20no_random_missing_time = speed_feature.copy()
    train_speed_data40no_random_missing_time = speed_feature.copy()
    train_speed_data60no_random_missing_time = speed_feature.copy()
    train_speed_data80no_random_missing_time = speed_feature.copy()
    train_speed_data20no_random_missing_time[rand < 0.2] = 0
    train_speed_data40no_random_missing_time[rand < 0.4] = 0
    train_speed_data60no_random_missing_time[rand < 0.6] = 0
    train_speed_data80no_random_missing_time[rand < 0.8] = 0
    train_mask20no_random_missing_time = mask.copy()
    train_mask40no_random_missing_time = mask.copy()
    train_mask60no_random_missing_time = mask.copy()
    train_mask80no_random_missing_time = mask.copy()
    train_mask20no_random_missing_time[rand < 0.2] = 0
    train_mask40no_random_missing_time[rand < 0.4] = 0
    train_mask60no_random_missing_time[rand < 0.6] = 0
    train_mask80no_random_missing_time[rand < 0.8] = 0
    # shape为（14，288，4248）
    np.savez(r'./data/all_train_data20no_random_missing_time', train_speed_data20no_random_missing_time,
             train_mask20no_random_missing_time)
    np.savez(r'./data/all_train_data40no_random_missing_time', train_speed_data40no_random_missing_time,
             train_mask40no_random_missing_time)
    np.savez(r'./data/all_train_data60no_random_missing_time', train_speed_data60no_random_missing_time,
             train_mask60no_random_missing_time)
    np.savez(r'./data/all_train_data80no_random_missing_time', train_speed_data80no_random_missing_time,
             train_mask80no_random_missing_time)

    # 按路段进行非随机的缺失，某一天的某个路段的数据全部缺失。
    rand2 = np.random.random((speed_feature.shape[0], speed_feature.shape[-1]))
    train_speed_data20no_random_missing_road = speed_feature.copy()
    train_speed_data40no_random_missing_road = speed_feature.copy()
    train_speed_data60no_random_missing_road = speed_feature.copy()
    train_speed_data80no_random_missing_road = speed_feature.copy()
    train_mask20no_random_missing_road = mask.copy()
    train_mask40no_random_missing_road = mask.copy()
    train_mask60no_random_missing_road = mask.copy()
    train_mask80no_random_missing_road = mask.copy()
    for i in range(speed_feature.shape[0]):
        for j in range(speed_feature.shape[-1]):
            if rand2[i, j] < 0.2:
                train_speed_data20no_random_missing_road[i, :, j] = 0.
                train_mask20no_random_missing_road[i, :, j] = False
    for i in range(speed_feature.shape[0]):
        for j in range(speed_feature.shape[-1]):
            if rand2[i, j] < 0.4:
                train_speed_data40no_random_missing_road[i, :, j] = 0.
                train_mask40no_random_missing_road[i, :, j] = False
    for i in range(speed_feature.shape[0]):
        for j in range(speed_feature.shape[-1]):
            if rand2[i, j] < 0.6:
                train_speed_data60no_random_missing_road[i, :, j] = 0.
                train_mask60no_random_missing_road[i, :, j] = False
    for i in range(speed_feature.shape[0]):
        for j in range(speed_feature.shape[-1]):
            if rand2[i, j] < 0.8:
                train_speed_data80no_random_missing_road[i, :, j] = 0.
                train_mask80no_random_missing_road[i, :, j] = False
    # print(np.sum(mask ^ train_mask20no_random_missing_road))
    # print(np.sum(mask ^ train_mask40no_random_missing_road))
    # print(np.sum(mask ^ train_mask60no_random_missing_road))
    # print(np.sum(mask ^ train_mask80no_random_missing_road))
    # 2000905
    # 4008673
    # 6013308
    # 8052195
    # shape为（14，288，4248）
    np.savez(r'./data/all_train_data20no_random_missing_road', train_speed_data20no_random_missing_road,
             train_mask20no_random_missing_road)
    np.savez(r'./data/all_train_data40no_random_missing_road', train_speed_data40no_random_missing_road,
             train_mask40no_random_missing_road)
    np.savez(r'./data/all_train_data60no_random_missing_road', train_speed_data60no_random_missing_road,
             train_mask60no_random_missing_road)
    np.savez(r'./data/all_train_data80no_random_missing_road', train_speed_data80no_random_missing_road,
             train_mask80no_random_missing_road)

# all_speed_train_data()
def read_speed_data():
    roadId_list, _ = get_roadId_list_and_adj_mat()  # 道路id能保证不重复
    # print(len(np.unique(roadId_list)))
    speed_data_path = r'./data/data_category_by_date/2018-04-07'
    dic = defaultdict(list)
    speed_feature = np.zeros((288, 4248, 1))
    # count_not_in_roadId_list=0
    with open(speed_data_path, 'r') as f:
        for line in f.readlines():
            if line is not None:
                line = line.split(' ')
                road_id, period, speed_val = int(line[0]), int(line[1]), float(line[2])
                dic[period].append([road_id, speed_val])
    for idx, item in dic.items():
        for i in item:

            pos = np.where(roadId_list == i[0])
            if len(pos[0]) != 1:
                continue
                # print('error in roadId_list')
                # print(i[0])

                # 用于统计有多少条数据的道路id是不在道路列表中的，统计结果显示2018-04-07有1778条数据不合规
                # count_not_in_roadId_list+=1
            else:
                speed_feature[idx - 1][pos[0][0]][0] = i[1]
    mask = speed_feature != 0  # 速度特征的掩码，有数据的地方为True，没数据的地方为False
    # print(count_not_in_roadId_list)
    # 两个返回值的shape都是(288,4248,1)
    # 这里返回的掩码，是最原始的有数据的位置的掩码

    # 标准化的步骤不应该在这里，先注释掉
    # mean = np.mean(speed_feature)
    # std = np.std(speed_feature)
    # print('mean:{},std:{}'.format(mean, std))
    # # mean:8.075717871387504,std:5.722213607431153
    # speed_feature = np.divide(np.subtract(speed_feature,mean),std)
    return speed_feature, mask


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_road_feature():
    feature_path = r'./data/路段属性数据20210428103734895408/路段属性数据_2920000403588.csv'
    feature_data = pd.read_csv(feature_path)

    road_length = feature_data['ROADLENGTH']  # 得到道路长度特征
    road_length_feature = road_length.div(road_length.max())  # 对道路长度进行归一化处理
    road_length_feature = np.expand_dims(road_length_feature, axis=-1)  # 最后面增加一个维度，用于和其他特征做拼接

    dir_name = feature_data['DIRNAME']  # 获取方向特征，北-南，南-北，东-西，西东
    class_mapping = {cla: ind for ind, cla in enumerate(np.unique(dir_name))}  # 构造映射字典
    dir_name = dir_name.map(class_mapping)
    dir_feature = to_categorical(dir_name, len(class_mapping))  # 得到方向特征的独热编码

    type_feature = feature_data['TYPE']
    type_mapping = {t: ind for ind, t in enumerate(np.unique(type_feature))}  # 构造道路类型到序号的映射字典
    type_feature = to_categorical(type_feature.map(type_mapping), len(type_mapping))
    # print( type_feature[80:96])

    jam_speed = feature_data['JAM_SPEED']
    jam_speed_feature = jam_speed.div(jam_speed.max())  # 获得拥堵速度特征并进行归一化
    jam_speed_feature = np.expand_dims(jam_speed_feature, axis=-1)

    # print(road_length_feature.shape,dir_feature.shape,type_feature.shape,jam_speed_feature.shape)
    road_feature = np.concatenate([road_length_feature, dir_feature, type_feature, jam_speed_feature], axis=1)
    # print(feature[:10])

    # 返回的道路特征序列的顺序和邻接矩阵中的道路序列对应
    # shape为 (4248, 10)
    return road_feature


def get_feature(loss_rate):
    road_feature = get_road_feature()  # 获得道路特征，shape为 (4248, 10)，每条道路的特征不同
    road_feature_shape = road_feature.shape
    speed_feature, true_mask = read_speed_data()  # 获得速度特征，两个返回值的shape都是(288,4248,1)

    train_mask = np.where(true_mask, np.random.rand(*true_mask.shape) < 1 - loss_rate, true_mask)
    train_speed_feature = np.where(train_mask, speed_feature, 0.)
    # tf.tile函数将按照轴，将道路特征复制[288,1]份，shape变为(4248*288,10*1)
    # 再经过reshape，将转换成（288，4248，10）
    road_feature = tf.reshape(tf.tile(road_feature, [288, 1]), (-1, road_feature_shape[0], road_feature_shape[1]))
    feature = tf.concat([speed_feature, road_feature], axis=-1)  # 按最后一个轴拼接

    train_feature = tf.concat([train_speed_feature, road_feature], axis=-1)
    # shape为（288，4248，11）,(288,4248,1), （288，4248，11）,,(288,4248,1),(288,4248,1)
    # feature，true_mask 是能得到的尽量完整的数据的特征和对应的速度的掩码，shape分别为（288，4248，11），（288，4248，1）
    # train_feature，train_mask 是在上面的基础上再丢掉一部分的数据特征和对应速度掩码，shape和上面相同。稀疏程度不同。
    # speed_feature, train_speed_feature 是只包含速度的特征，shape都是（288，4248，1）,后者更稀疏。
    np.savez('feature', feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature)
    return feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature


def get_all_feature():
    road_feature = get_road_feature()
    true_all_data = np.load('./all_speed_data.npz')
    # 保存的shape为（14，288，4248，1）
    true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    # 保存的shape为（14，288，4248,1），需要squeeze一下
    # true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature), np.squeeze(true_all_mask)

    # 处理时间特征
    week = np.array([4., 5., 6., 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3])
    time_slice = np.array(range(288))
    week_ten = np.repeat(week, 288 * 4248, axis=-1)
    week_ten = np.reshape(week_ten, (14, 288, 4248, 1))
    time_slice_ten = np.reshape(np.repeat(np.tile(time_slice, 14), 4248, axis=0), (14, 288, 4248, 1))

    # 以下shape均为（14，288，4248）
    data20miss = np.load("./data/all_train_data20random_missing.npz")
    train_speed_data20miss, train_mask20miss = data20miss['arr_0'], data20miss['arr_1']
    data40miss = np.load("./data/all_train_data40random_missing.npz")
    train_speed_data40miss, train_mask40miss = data40miss['arr_0'], data40miss['arr_1']
    data60miss = np.load("./data/all_train_data60random_missing.npz")
    train_speed_data60miss, train_mask60miss = data60miss['arr_0'], data60miss['arr_1']
    data80miss = np.load("./data/all_train_data80random_missing.npz")
    train_speed_data80miss, train_mask80miss = data80miss['arr_0'], data80miss['arr_1']

    train_speed_data20miss = np.expand_dims(train_speed_data20miss, -1)
    train_speed_data40miss = np.expand_dims(train_speed_data40miss, -1)
    train_speed_data60miss = np.expand_dims(train_speed_data60miss, -1)
    train_speed_data80miss = np.expand_dims(train_speed_data80miss, -1)

    road_feature = np.reshape(np.tile(road_feature, [14 * 288, 1]), (14, 288, 4248, -1))
    train_feature20miss = np.concatenate([train_speed_data20miss, road_feature, week_ten, time_slice_ten], axis=-1)
    train_feature40miss = np.concatenate([train_speed_data40miss, road_feature, week_ten, time_slice_ten], axis=-1)
    train_feature60miss = np.concatenate([train_speed_data60miss, road_feature, week_ten, time_slice_ten], axis=-1)
    train_feature80miss = np.concatenate([train_speed_data80miss, road_feature, week_ten, time_slice_ten], axis=-1)

    # 这里存储的是14天内所有速度数据，并且包含其他的固有特征，固有特征数量是10，加上速度特征就是11，经过了随机再稀疏，shape为（14，288，4248，11）
    np.save(r'./data/train_feature20random_missing', train_feature20miss)
    np.save(r'./data/train_feature40random_missing', train_feature40miss)
    np.save(r'./data/train_feature60random_missing', train_feature60miss)
    np.save(r'./data/train_feature80random_missing', train_feature80miss)

    # 处理按天的非随机缺失
    data20no_random_missing_time = np.load("./data/all_train_data20no_random_missing_time.npz")
    train_speed_data20no_random_missing_time = data20no_random_missing_time['arr_0']
    # train_mask20no_random_missing_time = data20no_random_missing_time['arr_1']
    data40no_random_missing_time = np.load("./data/all_train_data20no_random_missing_time.npz")
    train_speed_data40no_random_missing_time = data40no_random_missing_time['arr_0']
    # train_mask40no_random_missing_time = data40no_random_missing_time['arr_1']
    data60no_random_missing_time = np.load("./data/all_train_data20no_random_missing_time.npz")
    train_speed_data60no_random_missing_time = data60no_random_missing_time['arr_0']
    # train_mask60no_random_missing_time = data60no_random_missing_time['arr_1']
    data80no_random_missing_time = np.load("./data/all_train_data20no_random_missing_time.npz")
    train_speed_data80no_random_missing_time = data80no_random_missing_time['arr_0']
    # train_mask80no_random_missing_time = data80no_random_missing_time['arr_1']
    train_speed_data20no_random_missing_time = np.expand_dims(train_speed_data20no_random_missing_time, -1)
    train_speed_data40no_random_missing_time = np.expand_dims(train_speed_data40no_random_missing_time, -1)
    train_speed_data60no_random_missing_time = np.expand_dims(train_speed_data60no_random_missing_time, -1)
    train_speed_data80no_random_missing_time = np.expand_dims(train_speed_data80no_random_missing_time, -1)
    train_feature20no_random_missing_time = np.concatenate(
        [train_speed_data20no_random_missing_time, road_feature, week_ten, time_slice_ten], axis=-1)
    train_feature40no_random_missing_time = np.concatenate(
        [train_speed_data40no_random_missing_time, road_feature, week_ten, time_slice_ten], axis=-1)
    train_feature60no_random_missing_time = np.concatenate(
        [train_speed_data60no_random_missing_time, road_feature, week_ten, time_slice_ten], axis=-1)
    train_feature80no_random_missing_time = np.concatenate(
        [train_speed_data80no_random_missing_time, road_feature, week_ten, time_slice_ten], axis=-1)
    np.save(r'./data/train_feature20no_random_missing_time', train_feature20no_random_missing_time)
    np.save(r'./data/train_feature40no_random_missing_time', train_feature40no_random_missing_time)
    np.save(r'./data/train_feature60no_random_missing_time', train_feature60no_random_missing_time)
    np.save(r'./data/train_feature80no_random_missing_time', train_feature80no_random_missing_time)

    # 按道路的非随机缺失
    data20no_random_missing_road = np.load("./data/all_train_data20no_random_missing_road.npz")
    train_speed_data20no_random_missing_road = data20no_random_missing_road['arr_0']
    # train_mask20no_random_missing_road = data20no_random_missing_road['arr_1']
    data40no_random_missing_road = np.load("./data/all_train_data20no_random_missing_road.npz")
    train_speed_data40no_random_missing_road = data40no_random_missing_road['arr_0']
    # train_mask40no_random_missing_road = data40no_random_missing_road['arr_1']
    data60no_random_missing_road = np.load("./data/all_train_data20no_random_missing_road.npz")
    train_speed_data60no_random_missing_road = data60no_random_missing_road['arr_0']
    # train_mask60no_random_missing_road = data60no_random_missing_road['arr_1']
    data80no_random_missing_road = np.load("./data/all_train_data20no_random_missing_road.npz")
    train_speed_data80no_random_missing_road = data80no_random_missing_road['arr_0']
    # train_mask80no_random_missing_road = data80no_random_missing_road['arr_1']
    train_speed_data20no_random_missing_road = np.expand_dims(train_speed_data20no_random_missing_road, -1)
    train_speed_data40no_random_missing_road = np.expand_dims(train_speed_data40no_random_missing_road, -1)
    train_speed_data60no_random_missing_road = np.expand_dims(train_speed_data60no_random_missing_road, -1)
    train_speed_data80no_random_missing_road = np.expand_dims(train_speed_data80no_random_missing_road, -1)
    train_feature20no_random_missing_road = np.concatenate(
        [train_speed_data20no_random_missing_road, road_feature, week_ten, time_slice_ten],
        axis=-1)
    train_feature40no_random_missing_road = np.concatenate(
        [train_speed_data40no_random_missing_road, road_feature, week_ten, time_slice_ten],
        axis=-1)
    train_feature60no_random_missing_road = np.concatenate(
        [train_speed_data60no_random_missing_road, road_feature, week_ten, time_slice_ten],
        axis=-1)
    train_feature80no_random_missing_road = np.concatenate(
        [train_speed_data80no_random_missing_road, road_feature, week_ten, time_slice_ten],
        axis=-1)
    np.save(r'./data/train_feature20no_random_missing_road', train_feature20no_random_missing_road)
    np.save(r'./data/train_feature40no_random_missing_road', train_feature40no_random_missing_road)
    np.save(r'./data/train_feature60no_random_missing_road', train_feature60no_random_missing_road)
    np.save(r'./data/train_feature80no_random_missing_road', train_feature80no_random_missing_road)

# get_all_feature()

def gen_weekday_weekend_data():
    true_all_data = np.load('./all_speed_data.npz')
    true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    # 保存的shape为（14，288，4248,1），需要squeeze一下
    true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature), np.squeeze(true_all_mask)
    weekday_all_data = true_all_speed_feature[[0,3,4,5,6,7,10,11,12,13]]
    mask_weekday_all_data = true_all_mask[[0, 3, 4, 5, 6, 7, 10, 11, 12, 13]]
    np.save(r'./data/weekday/weekday_all_data', weekday_all_data)
    np.save(r'./data/weekday/mask_weekday_all_data', mask_weekday_all_data)
    weekend_all_data = true_all_speed_feature[[1,2,8,9]]
    mask_weekend_all_data = true_all_mask[[1,2,8,9]]
    np.save(r'./data/weekend/weekend_all_data', weekend_all_data)
    np.save(r'./data/weekend/mask_weekend_all_data', mask_weekend_all_data)
    # 以下shape均为（14，288，4248）, 随机缺失
    data20miss = np.load("./data/all_train_data20random_missing.npz")
    train_speed_data20miss, train_mask20miss = data20miss['arr_0'], data20miss['arr_1']
    data40miss = np.load("./data/all_train_data40random_missing.npz")
    train_speed_data40miss, train_mask40miss = data40miss['arr_0'], data40miss['arr_1']
    data60miss = np.load("./data/all_train_data60random_missing.npz")
    train_speed_data60miss, train_mask60miss = data60miss['arr_0'], data60miss['arr_1']
    data80miss = np.load("./data/all_train_data80random_missing.npz")
    train_speed_data80miss, train_mask80miss = data80miss['arr_0'], data80miss['arr_1']
    weekday20miss, mask_weekday20miss = train_speed_data20miss[[0,3,4,5,6,7,10,11,12,13]],train_mask20miss[[0,3,4,5,6,7,10,11,12,13]]
    weekday40miss, mask_weekday40miss = train_speed_data40miss[[0,3,4,5,6,7,10,11,12,13]],train_mask40miss[[0,3,4,5,6,7,10,11,12,13]]
    weekday60miss, mask_weekday60miss = train_speed_data60miss[[0,3,4,5,6,7,10,11,12,13]],train_mask60miss[[0,3,4,5,6,7,10,11,12,13]]
    weekday80miss, mask_weekday80miss = train_speed_data80miss[[0,3,4,5,6,7,10,11,12,13]],train_mask80miss[[0,3,4,5,6,7,10,11,12,13]]

    weekend20miss, mask_weekend20miss = train_speed_data20miss[[1,2,8,9]],train_mask20miss[[1,2,8,9]]
    weekend40miss, mask_weekend40miss = train_speed_data40miss[[1,2,8,9]],train_mask40miss[[1,2,8,9]]
    weekend60miss, mask_weekend60miss = train_speed_data60miss[[1,2,8,9]],train_mask60miss[[1,2,8,9]]
    weekend80miss, mask_weekend80miss = train_speed_data80miss[[1,2,8,9]],train_mask80miss[[1,2,8,9]]
    np.save(r'./data/weekday/weekday20miss',weekday20miss)
    np.save(r'./data/weekday/weekday40miss',weekday40miss)
    np.save(r'./data/weekday/weekday60miss',weekday60miss)
    np.save(r'./data/weekday/weekday80miss',weekday80miss)
    np.save(r'./data/weekday/mask_weekday20miss',mask_weekday20miss)
    np.save(r'./data/weekday/mask_weekday40miss',mask_weekday40miss)
    np.save(r'./data/weekday/mask_weekday60miss',mask_weekday60miss)
    np.save(r'./data/weekday/mask_weekday80miss',mask_weekday80miss)
    np.save(r'./data/weekend/mask_weekend20miss',mask_weekend20miss)
    np.save(r'./data/weekend/mask_weekend40miss',mask_weekend40miss)
    np.save(r'./data/weekend/mask_weekend60miss',mask_weekend60miss)
    np.save(r'./data/weekend/mask_weekend80miss',mask_weekend80miss)
    np.save(r'./data/weekend/weekend20miss',weekend20miss)
    np.save(r'./data/weekend/weekend40miss',weekend40miss)
    np.save(r'./data/weekend/weekend60miss',weekend60miss)
    np.save(r'./data/weekend/weekend80miss',weekend80miss)

def gen_weekday_weekend_data_no_random():
    # true_all_data = np.load('./all_speed_data.npz')
    # true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    # 保存的shape为（14，288，4248,1），需要squeeze一下
    # true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature), np.squeeze(true_all_mask)
    # weekday_all_data = true_all_speed_feature[[0,3,4,5,6,7,10,11,12,13]]
    # mask_weekday_all_data = true_all_mask[[0, 3, 4, 5, 6, 7, 10, 11, 12, 13]]
    # np.save(r'./data/weekday/weekday_all_data', weekday_all_data)
    # np.save(r'./data/weekday/mask_weekday_all_data', mask_weekday_all_data)
    # weekend_all_data = true_all_speed_feature[[1,2,8,9]]
    # mask_weekend_all_data = true_all_mask[[1,2,8,9]]
    # np.save(r'./data/weekend/weekend_all_data', weekend_all_data)
    # np.save(r'./data/weekend/mask_weekend_all_data', mask_weekend_all_data)
    # 以下shape均为（14，288，4248）, 非随机缺失
    data20miss = np.load("./data/all_train_data20no_random_missing_road.npz")
    train_speed_data20miss, train_mask20miss = data20miss['arr_0'], data20miss['arr_1']
    data40miss = np.load("./data/all_train_data40no_random_missing_road.npz")
    train_speed_data40miss, train_mask40miss = data40miss['arr_0'], data40miss['arr_1']
    data60miss = np.load("./data/all_train_data60no_random_missing_road.npz")
    train_speed_data60miss, train_mask60miss = data60miss['arr_0'], data60miss['arr_1']
    data80miss = np.load("./data/all_train_data80no_random_missing_road.npz")
    train_speed_data80miss, train_mask80miss = data80miss['arr_0'], data80miss['arr_1']
    weekday20miss, mask_weekday20miss = train_speed_data20miss[[0,3,4,5,6,7,10,11,12,13]],train_mask20miss[[0,3,4,5,6,7,10,11,12,13]]
    weekday40miss, mask_weekday40miss = train_speed_data40miss[[0,3,4,5,6,7,10,11,12,13]],train_mask40miss[[0,3,4,5,6,7,10,11,12,13]]
    weekday60miss, mask_weekday60miss = train_speed_data60miss[[0,3,4,5,6,7,10,11,12,13]],train_mask60miss[[0,3,4,5,6,7,10,11,12,13]]
    weekday80miss, mask_weekday80miss = train_speed_data80miss[[0,3,4,5,6,7,10,11,12,13]],train_mask80miss[[0,3,4,5,6,7,10,11,12,13]]

    weekend20miss, mask_weekend20miss = train_speed_data20miss[[1,2,8,9]],train_mask20miss[[1,2,8,9]]
    weekend40miss, mask_weekend40miss = train_speed_data40miss[[1,2,8,9]],train_mask40miss[[1,2,8,9]]
    weekend60miss, mask_weekend60miss = train_speed_data60miss[[1,2,8,9]],train_mask60miss[[1,2,8,9]]
    weekend80miss, mask_weekend80miss = train_speed_data80miss[[1,2,8,9]],train_mask80miss[[1,2,8,9]]
    np.save(r'./data/weekday/weekday20no_random_miss',weekday20miss)
    np.save(r'./data/weekday/weekday40no_random_miss',weekday40miss)
    np.save(r'./data/weekday/weekday60no_random_miss',weekday60miss)
    np.save(r'./data/weekday/weekday80no_random_miss',weekday80miss)
    np.save(r'./data/weekday/mask_weekday20no_random_miss',mask_weekday20miss)
    np.save(r'./data/weekday/mask_weekday40no_random_miss',mask_weekday40miss)
    np.save(r'./data/weekday/mask_weekday60no_random_miss',mask_weekday60miss)
    np.save(r'./data/weekday/mask_weekday80no_random_miss',mask_weekday80miss)
    np.save(r'./data/weekend/mask_weekend20no_random_miss',mask_weekend20miss)
    np.save(r'./data/weekend/mask_weekend40no_random_miss',mask_weekend40miss)
    np.save(r'./data/weekend/mask_weekend60no_random_miss',mask_weekend60miss)
    np.save(r'./data/weekend/mask_weekend80no_random_miss',mask_weekend80miss)
    np.save(r'./data/weekend/weekend20no_random_miss',weekend20miss)
    np.save(r'./data/weekend/weekend40no_random_miss',weekend40miss)
    np.save(r'./data/weekend/weekend60no_random_miss',weekend60miss)
    np.save(r'./data/weekend/weekend80no_random_miss',weekend80miss)

# gen_weekday_weekend_data()
# gen_weekday_weekend_data_no_random()
# def load_data(dataset_str):
#     """Load data."""
#     FILE_PATH = os.path.abspath(__file__)
#     DIR_PATH = os.path.dirname(FILE_PATH)
#     DATA_PATH = os.path.join(DIR_PATH, 'data/')
#
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATA_PATH, dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder),
#                                     max(test_idx_reorder) + 1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range - min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range - min(test_idx_range), :] = ty
#         ty = ty_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#
#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y) + 500)
#
#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])
#
#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]
#
#     # ***********
#
#
#     tmp = np.load('./data/roadsectId_list_and_adj_mat.npz')
#
#     # 道路id的列表，索引与邻接矩阵中的索引相对应
#     # array([  11301,   11302,   11303, ..., 3519122, 3519201, 3519202], dtype=int64)
#     roadsectId_list=tmp['arr_0']
#     adj_mat= tmp['arr_1']   #邻接矩阵
#
#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
#
#


# t=get_feature(0.2)
# print(t[1][1])
# a,b=get_roadId_list_and_adj_mat()
# print(type(b),b.shape)
# print(1-b)
# speed_feature, true_mask = read_speed_data()
# print(speed_feature.shape)
# speed=read_speed_data4predict()
# print(speed.shape)
# print(speed[1][1])
# read_all_speed_data()
# speed,mask = read_speed_data()
# np.save('speed_feature20180407',speed)
# all_speed_train_data()
# get_all_feature()
# save_sparse_adj()
# gen_weekday_weekend_data()

def tmp_func():
    road_feature = get_road_feature()
    # road_feature = np.array(road_feature)
    true_all_data = np.load('./all_speed_data.npz')
    # 保存的shape为（14，288，4248，1）
    true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    week = np.array([4., 5., 6., 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3])
    time_slice = np.array(range(288))
    week_ten = np.repeat(week, 288 * 4248, axis=-1)
    week_ten = np.reshape(week_ten, (14, 288, 4248, 1))
    time_slice_ten = np.reshape(np.repeat(np.tile(time_slice, 14), 4248, axis=0), (14, 288, 4248, 1))
    tmp = np.tile(road_feature, [4032, 1])      # numpy版本导致问题，
    road_feature = np.reshape(tmp, (14, 288, 4248, -1))
    all_feature = np.concatenate([true_all_speed_feature, road_feature, week_ten, time_slice_ten], axis=-1)
    print(all_feature.shape)
    np.save(r'./data/all_feature', all_feature)
    # 这里存储的是速度加上其他的特征，14天，shape（14,288,4248,13)

# tmp_func()
# gen_weekday_weekend_data_no_random()

def select_500_road():
    init_road_index = 10      # 0
    roadsecId_list_and_adj_mat = np.load('./data/roadsectId_list_and_adj_mat.npz')
    roadsecId_list = roadsecId_list_and_adj_mat['arr_0']
    adj_mat = roadsecId_list_and_adj_mat['arr_1']
    selected_road_list = []
    tmp_list = deque()
    tmp_list.append(init_road_index)
    selected_road_cnt = 0

    while len(tmp_list) != 0 and selected_road_cnt < 500:
        i = tmp_list.popleft()
        if i in selected_road_list:
            continue
        for j in range(len(adj_mat[i])):
            if adj_mat[i][j] == 1:
                tmp_list.append(j)
        selected_road_list.append(i)
        selected_road_cnt += 1
    # print(len(selected_road_list))
    # print(selected_road_list[:10])
    selected_road_mat = adj_mat[selected_road_list][:, selected_road_list]
    # print(adj_mat.shape)
    # print(selected_road_mat.shape,np.sum(selected_road_mat))
    # 这里保存的是随机选择的100条道路，这100条道路尽可能连接
    # 前面是一个列表，这个列表中存储的是对于原来4248条道路的索引，300为初始选中的第一条道路时，前10条道路在原来4248条道路的索引为
    # [300, 301, 3791, 302, 2254, 2258, 266, 270, 3792, 3802]
    # 后面一个是矩阵，矩阵shape为100*100，只包含0，1 。
    np.savez("./data/selected500road_index_and_adj_mat", selected_road_list, selected_road_mat)

# select_500_road()

def gen_feature_of_edge():
    true_all_data = np.load('./all_speed_data.npz')
    # 保存的shape为（14，288，4248，1）
    true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    true_all_speed_feature = np.reshape(np.squeeze(true_all_speed_feature), [14*288,4248]) # shape (14*288,4248)
    cnt = np.count_nonzero(true_all_speed_feature, axis=0)  # shape (,4248)
    sum_ = np.sum(true_all_speed_feature, axis=0)   # shape(,4248)
    avg = np.divide(sum_, cnt)  #shape (,4248)

    roadsecId_list_and_adj_mat = np.load('./data/roadsectId_list_and_adj_mat.npz')
    roadsecId_list = roadsecId_list_and_adj_mat['arr_0']
    adj_mat = roadsecId_list_and_adj_mat['arr_1']

    res=np.zeros([4248,4248])

    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[i])):
            if adj_mat[i][j]==1:
                count = 0.00001     # 避免除以0
                same = 0
                for k in range(14*288):
                    if true_all_speed_feature[k,i]!=0 and true_all_speed_feature[k,j]!=0:
                        if (true_all_speed_feature[k,i]>=avg[i] and true_all_speed_feature[k,j]>=avg[j]) or \
                                (true_all_speed_feature[k,i]<avg[i] and true_all_speed_feature[k,j]<avg[j]):
                            same += 1
                        count += 1
                res[i][j] = np.divide(same,count)
    np.save("./data/feature_of_edge",res)

# gen_feature_of_edge()
# print("over! ")

def select_big_small_road_from_500_roads():
    selected500road_list_and_adj_mat = np.load("./data/selected500road_index_and_adj_mat.npz")
    roadsectId_list = selected500road_list_and_adj_mat['arr_0']  # 被选中的道路在原先4248条道路中的编号
    roadsect_adj_mat = selected500road_list_and_adj_mat['arr_1']  # 被选中的道路的邻接矩阵

    true_all_data = np.load('./all_speed_data.npz')  # (14,288,4248,1)
    # shape(14,288,4248,1)
    true_all_speed_data, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
    true_all_speed_data500 = np.squeeze(true_all_speed_data[:, :, roadsectId_list, :])  # shape(14,288,500,1)
    true_all_mask500 = np.squeeze(true_all_mask[:, :, roadsectId_list, :])



