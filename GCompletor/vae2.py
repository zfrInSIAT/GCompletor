import time
import numpy as np
import tensorflow as tf
# import tensorflow
from tensorflow.contrib.slim import fully_connected as fc
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# import matplotlib.pyplot as plt
# %matplotlib inline
import sys
# tf.disable_eager_execution()
sys.path.append('../baselines')
from baselines.metrics import mape, rmse
from utils import get_deviation

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# num_sample = mnist.train.num_examples
# input_dim = mnist.train.images[0].shape[0]


class VariantionalAutoencoder(object):

    def __init__(self,input_dim, learning_rate=1e-4, batch_size=64, n_z=16):  # learning_rate=1e-4
        # Set hyperparameters
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        # Build the graph
        self.build()

        # Initialize paramters
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(
            name='x', dtype=tf.float32, shape=[None, self.input_dim])
        # Encode
        # tmp_ = tf.nn.dropout(self.x, rate=0.3)
        # x -> z_mean, z_sigma -> z
        # f1 = fc(self.x, 256, scope='enc_fc1', activation_fn=tf.nn.relu)
        # f2 = fc(f1, 128, scope='enc_fc2', activation_fn=tf.nn.relu)
        # f3 = fc(f2, 64, scope='enc_fc3', activation_fn=tf.nn.relu)
        # f1 = fc(tmp_, 256, scope='enc_fc1', activation_fn=tf.nn.relu)
        do = tf.nn.dropout(self.x,rate=0.2)
        # f1 = fc(self.x, 512, scope='enc_fc1', activation_fn=tf.nn.relu)
        f1 = fc(do, 512, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 256, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 128, scope='enc_fc3', activation_fn=tf.nn.relu)
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu',
                       activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma',
                                 activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(self.z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 128, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 256, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.relu)
        self.x_hat = fc(g3, self.input_dim, scope='dec_fc4',
                        activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10

        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon + self.x_hat) +
            (1 - self.x) * tf.log(epsilon + 1 - self.x_hat),
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between
        # the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) -
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = self.recon_loss  + self.latent_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)

        self.losses = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_loss,
        }
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.losses],
            feed_dict={self.x: x}
        )
        return losses

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


def trainer(model_class, input_dim, train_data, learning_rate=0.000001,
            batch_size=10, num_epoch=700, n_z=16, log_step=5):        # 700
    # vae mape:  35.04757759101026  rmse:  2.516632172941658        50轮
    # 34.13594542330274  rmse:  2.4458262870458998 300
    # Create a model
    model = model_class(input_dim=input_dim,
        learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)

    # Training loop
    for epoch in range(num_epoch):
        start_time = time.time()

        # Run an epoch
        for iter in range(10 // batch_size):
            # Get a batch
            # batch是一个二元组，0是（64，784），1是（64，10）也就是标签
            # batch = mnist.train.next_batch(batch_size)
            # Execute the forward and backward pass
            # Report computed loss
            losses = model.run_single_step(train_data[np.random.choice(train_data.shape[0],train_data.shape[0],replace=False)])
            # losses = model.run_single_step(
            #     np.multiply(true_all_speed_feature_mat[np.random.choice(9, batch_size, replace=False)],
            #                 np.where(np.random.random((batch_size, 4248 * 288)) < 0.2, 0., 1.)))

        end_time = time.time()

        # Log the loss
        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            print(log_str)

    print('Done!')
    return model

def get_mape_rmse(input_dim,train_data, test_true_data_and_mask, test_data_and_test_data_mask_list):
    # 第1个参数是vae的输入特征维度。
    # 第2个参数是训练数据，是个张量，
    # 第3个参数是用于测试的真实值和其掩码的列表，是len为2的列表
    # 第4个参数是测试的数据和其掩码的列表，型为[[t1,m1],[t2,m2],...]
    # 第5个参数的数据和第三个参数中的数据同型
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    model_vae = trainer(VariantionalAutoencoder, input_dim, train_data_scaled)

    for test_data, test_data_mask in test_data_and_test_data_mask_list:
        predict_data_scaled = model_vae.reconstructor(scaler.transform(test_data))
        predict_data = scaler.inverse_transform(predict_data_scaled)
        mask = test_data_mask ^ test_true_data_and_mask[1]
        mape_res = mape(test_true_data_and_mask[0][mask], predict_data[mask])
        rmse_res = rmse(test_true_data_and_mask[0][mask], predict_data[mask])
        print("vae mape: ",mape_res," rmse: ",rmse_res)
        true_distribution, pret_distribution = get_deviation(test_true_data_and_mask[0], test_true_data_and_mask[1],predict_data)
        print("true_distribution: ", true_distribution)
        print("pret_distribution: ", pret_distribution)

        sorted_list = np.load(r'./data/sorted_var_500_road.npy')
        a = np.reshape(test_true_data_and_mask[0],[-1,288,500])[...,sorted_list[400:]]
        b = np.reshape(predict_data,[-1,288,500])[...,sorted_list[400:]]
        m = np.reshape(mask,[-1,288,500])[...,sorted_list[400:]]
        BTRMF_res2_mape2_big_var = mape(a[m], b[m])
        BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
        c = np.reshape(test_true_data_and_mask[0],[-1,288,500])[...,sorted_list[:400]]
        d = np.reshape(predict_data,[-1,288,500])[...,sorted_list[:400]]
        e = np.reshape(mask,[-1,288,500])[...,sorted_list[:400]]
        BTRMF_res2_mape2_small_var = mape(c[e], d[e])
        BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])

        print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
        print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

        deviation_true_data, deviation_pret_data = get_deviation(test_true_data_and_mask[0], test_true_data_and_mask[1], predict_data)
        print("BTRMF_deviation_pret_data", deviation_pret_data)
        print("deviation_true_data", deviation_true_data)

        speed_feature_ten = np.reshape(test_true_data_and_mask[0], [-1, 288,500])
        data_ten = np.reshape(predict_data, [-1, 288,500])
        mask_ten = np.reshape(mask, [-1, 288,500])
        str2 = "分时段\n"
        for item in [[0, 48], [48, 96], [96, 144], [144, 192], [192, 240], [240, 288]]:
            t1=mask_ten[:, item[0]:item[1],:]
            t2=speed_feature_ten[:, item[0]:item[1],:][t1]
            t3=data_ten[:, item[0]:item[1],:][t1]
            if len(t2) == 0:
                mean_res_mape1 = 99999
                mean_res_rmse1 = 99999
            else:
                mean_res_mape1 = mape(t2,t3)
                mean_res_rmse1 = rmse(t2,t3)
            str2 += "mape:{},rmse:{}\n".format(mean_res_mape1, mean_res_rmse1)
            # str2 += "mape:{}\n".format(mean_res_mape1)
        print(str2)

        # print("输出方差较大的10%的路段的补全效果")
        # big_var_road = np.load(r'./data/big_var_road.npy')
        # small_var_road = np.load(r'./data/small_var_road.npy')
        # # 这里需要特殊处理，把shape为（天数，288*4248）reshape一下，
        # a = test_true_data_and_mask[0].reshape([-1,288,4248])[...,big_var_road]
        # b = predict_data.reshape([-1,288,4248])[...,big_var_road]
        # m = mask.reshape([-1,288,4248])[...,big_var_road]
        # BTRMF_res2_mape2_big_var = mape(a[m], b[m])
        # BTRMF_res2_rmse2_big_var = rmse(a[m], b[m])
        # c = test_true_data_and_mask[0].reshape([-1,288,4248])[...,small_var_road]
        # d = predict_data.reshape([-1,288,4248])[...,small_var_road]
        # e = mask.reshape([-1,288,4248])[...,small_var_road]
        # BTRMF_res2_mape2_small_var = mape(c[e], d[e])
        # BTRMF_res2_rmse2_small_var = rmse(c[e], d[e])
        #
        # print("大方差的道路: mape: ", BTRMF_res2_mape2_big_var, "rmse: ", BTRMF_res2_rmse2_big_var)
        # print("小方差的道路: mape: ", BTRMF_res2_mape2_small_var, "rmse: ", BTRMF_res2_rmse2_small_var)

selected100road_list_and_adj_mat = np.load("./data/selected500road_index_and_adj_mat.npz")
roadsectId_list = selected100road_list_and_adj_mat['arr_0']
roadsect_adj_mat = selected100road_list_and_adj_mat['arr_1']

# 数据加载与处理
true_all_data = np.load('./all_speed_data.npz')
true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
# 保存的shape为（14，288，4248,1），需要squeeze一下
true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature)[...,roadsectId_list], np.squeeze(true_all_mask)[...,roadsectId_list]
# 以下shape均为（14，288，4248）, 随机缺失
data20miss = np.load("./data/all_train_data20random_missing.npz")
train_speed_data20miss, train_mask20miss = data20miss['arr_0'][...,roadsectId_list], data20miss['arr_1'][...,roadsectId_list]
data40miss = np.load("./data/all_train_data40random_missing.npz")
train_speed_data40miss, train_mask40miss = data40miss['arr_0'][...,roadsectId_list], data40miss['arr_1'][...,roadsectId_list]
data60miss = np.load("./data/all_train_data60random_missing.npz")
train_speed_data60miss, train_mask60miss = data60miss['arr_0'][...,roadsectId_list], data60miss['arr_1'][...,roadsectId_list]
data80miss = np.load("./data/all_train_data80random_missing.npz")
train_speed_data80miss, train_mask80miss = data80miss['arr_0'][...,roadsectId_list], data80miss['arr_1'][...,roadsectId_list]
#
true_all_speed_feature_mat = true_all_speed_feature.reshape(14, -1)
true_all_mask_mat = true_all_mask.reshape(14, -1)
train_speed_data20miss_mat = train_speed_data20miss.reshape(14, -1)
train_mask20miss_mat = train_mask20miss.reshape(14, -1)
train_speed_data40miss_mat = train_speed_data40miss.reshape(14, -1)
train_mask40miss_mat = train_mask40miss.reshape(14, -1)
train_speed_data60miss_mat = train_speed_data60miss.reshape(14, -1)
train_mask60miss_mat = train_mask60miss.reshape(14, -1)
train_speed_data80miss_mat = train_speed_data80miss.reshape(14, -1)
train_mask80miss_mat = train_mask80miss.reshape(14, -1)
# print("所有数据的随机缺失和非随机缺失")
# data20nr_miss = np.load("./data/all_train_data20no_random_missing_road.npz")
# train_speed_data20nr_miss_ten, train_mask20nr_miss_ten = data20nr_miss['arr_0'][...,roadsectId_list], data20nr_miss['arr_1'][...,roadsectId_list]
# data40nr_miss = np.load("./data/all_train_data40no_random_missing_road.npz")
# train_speed_data40nr_miss_ten, train_mask40nr_miss_ten = data40nr_miss['arr_0'][...,roadsectId_list], data40nr_miss['arr_1'][...,roadsectId_list]
# data60nr_miss = np.load("./data/all_train_data60no_random_missing_road.npz")
# train_speed_data60nr_miss_ten, train_mask60nr_miss_ten = data60nr_miss['arr_0'][...,roadsectId_list], data60nr_miss['arr_1'][...,roadsectId_list]
# data80nr_miss = np.load("./data/all_train_data80no_random_missing_road.npz")
# train_speed_data80nr_miss_ten, train_mask80nr_miss_ten = data80nr_miss['arr_0'][...,roadsectId_list], data80nr_miss['arr_1'][...,roadsectId_list]
# # 进行reshape
# train_speed_data20nr_miss_mat = train_speed_data20nr_miss_ten.reshape(14, -1)
# train_speed_data40nr_miss_mat = train_speed_data40nr_miss_ten.reshape(14, -1)
# train_speed_data60nr_miss_mat = train_speed_data60nr_miss_ten.reshape(14, -1)
# train_speed_data80nr_miss_mat = train_speed_data80nr_miss_ten.reshape(14, -1)
# train_mask20nr_miss_mat = train_mask20nr_miss_ten.reshape(14, -1)
# train_mask40nr_miss_mat = train_mask40nr_miss_ten.reshape(14, -1)
# train_mask60nr_miss_mat = train_mask60nr_miss_ten.reshape(14, -1)
# train_mask80nr_miss_mat = train_mask80nr_miss_ten.reshape(14, -1)
#
# get_mape_rmse(288*500,
#               true_all_speed_feature_mat[:10],
#               [true_all_speed_feature_mat[10:], true_all_mask_mat[10:]],
              # [[train_speed_data20miss_mat[10:], train_mask20miss_mat[10:]]])#,
               # [train_speed_data40miss_mat[10:], train_mask40miss_mat[10:]],
               # [train_speed_data60miss_mat[10:], train_mask60miss_mat[10:]],
               # [train_speed_data80miss_mat[10:], train_mask80miss_mat[10:]],
               # [[train_speed_data20nr_miss_mat[10:], train_mask20nr_miss_mat[10:]]])#,
               # [train_speed_data40nr_miss_mat[10:], train_mask40nr_miss_mat[10:]],
               # [train_speed_data60nr_miss_mat[10:], train_mask60nr_miss_mat[10:]],
               # [train_speed_data80nr_miss_mat[10:], train_mask80nr_miss_mat[10:]]])

# print("工作日的数据，前面是随机缺失的，后面是非随机缺失的")
# weekday_all_data = np.load(r'./data/weekday/weekday_all_data.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday_all_data = np.load(r'./data/weekday/mask_weekday_all_data.npy')[...,roadsectId_list].reshape([10, -1])
#
# weekday20miss = np.load(r'./data/weekday/weekday20miss.npy')[...,roadsectId_list].reshape([10, -1])
# weekday40miss = np.load(r'./data/weekday/weekday40miss.npy')[...,roadsectId_list].reshape([10, -1])
# weekday60miss = np.load(r'./data/weekday/weekday60miss.npy')[...,roadsectId_list].reshape([10, -1])
# weekday80miss = np.load(r'./data/weekday/weekday80miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday20miss = np.load(r'./data/weekday/mask_weekday20miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday40miss = np.load(r'./data/weekday/mask_weekday40miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday60miss = np.load(r'./data/weekday/mask_weekday60miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday80miss = np.load(r'./data/weekday/mask_weekday80miss.npy')[...,roadsectId_list].reshape([10, -1])
#
# weekday20nr_miss = np.load(r'./data/weekday/weekday20no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])
# weekday40nr_miss = np.load(r'./data/weekday/weekday40no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])
# weekday60nr_miss = np.load(r'./data/weekday/weekday60no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])
# weekday80nr_miss = np.load(r'./data/weekday/weekday80no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday20nr_miss = np.load(r'./data/weekday/mask_weekday20no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday40nr_miss = np.load(r'./data/weekday/mask_weekday40no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday60nr_miss = np.load(r'./data/weekday/mask_weekday60no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])
# mask_weekday80nr_miss = np.load(r'./data/weekday/mask_weekday80no_random_miss.npy')[...,roadsectId_list].reshape([10, -1])

# get_mape_rmse(288*500,
#               weekday_all_data[:7],
#               [weekday_all_data[7:], mask_weekday_all_data[7:]],
#               [[weekday20miss[7:], mask_weekday20miss[7:]],
#                # [weekday40miss[7:], mask_weekday40miss[7:]],
#                # [weekday60miss[7:], mask_weekday60miss[7:]],
#                # [weekday80miss[7:], mask_weekday80miss[7:]],
#                [weekday20nr_miss[7:], mask_weekday20nr_miss[7:]]])#,
#                # [weekday40nr_miss[7:], mask_weekday40nr_miss[7:]],
#                # [weekday60nr_miss[7:], mask_weekday60nr_miss[7:]],
#                # [weekday80nr_miss[7:], mask_weekday80nr_miss[7:]]])
# #
print("随机缺失，处理周末数据")
weekend_all_data = np.load(r'./data/weekend/weekend_all_data.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend_all_data = np.load(r'./data/weekend/mask_weekend_all_data.npy')[...,roadsectId_list].reshape([4, -1])

weekend20miss = np.load(r'./data/weekend/weekend20miss.npy')[...,roadsectId_list].reshape([4, -1])
weekend40miss = np.load(r'./data/weekend/weekend40miss.npy')[...,roadsectId_list].reshape([4, -1])
weekend60miss = np.load(r'./data/weekend/weekend60miss.npy')[...,roadsectId_list].reshape([4, -1])
weekend80miss = np.load(r'./data/weekend/weekend80miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend20miss = np.load(r'./data/weekend/mask_weekend20miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend40miss = np.load(r'./data/weekend/mask_weekend40miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend60miss = np.load(r'./data/weekend/mask_weekend60miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend80miss = np.load(r'./data/weekend/mask_weekend80miss.npy')[...,roadsectId_list].reshape([4, -1])

weekend20nr_miss = np.load(r'./data/weekend/weekend20no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
weekend40nr_miss = np.load(r'./data/weekend/weekend40no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
weekend60nr_miss = np.load(r'./data/weekend/weekend60no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
weekend80nr_miss = np.load(r'./data/weekend/weekend80no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend20nr_miss = np.load(r'./data/weekend/mask_weekend20no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend40nr_miss = np.load(r'./data/weekend/mask_weekend40no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend60nr_miss = np.load(r'./data/weekend/mask_weekend60no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
mask_weekend80nr_miss = np.load(r'./data/weekend/mask_weekend80no_random_miss.npy')[...,roadsectId_list].reshape([4, -1])
#
get_mape_rmse(288*500,
              weekend_all_data[:3],
              [weekend_all_data[3:], mask_weekend_all_data[3:]],
              [[weekend20miss[3:], mask_weekend20miss[3:]],
#                [weekend40miss[3:], mask_weekend40miss[3:]],
#                [weekend60miss[3:], mask_weekend60miss[3:]],
#                [weekend80miss[3:], mask_weekend80miss[3:]],
               [weekend20nr_miss[3:], mask_weekend20nr_miss[3:]]])# ,
#                [weekend40nr_miss[3:], mask_weekend40nr_miss[3:]],
#                [weekend60nr_miss[3:], mask_weekend60nr_miss[3:]],
#                [weekend80nr_miss[3:], mask_weekend80nr_miss[3:]]])




