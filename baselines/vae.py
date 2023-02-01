import time
import numpy as np
# from tf_slim import reset_default_graph
import tensorflow as tf
from tf_slim import fully_connected as fc

from sklearn.preprocessing import MinMaxScaler,StandardScaler
# import matplotlib.pyplot as plt
# %matplotlib inline
import sys

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
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        # tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(
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
        eps = tf.compat.v1.random_normal(
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
            self.x * tf.compat.v1.log(epsilon + self.x_hat) +
            (1 - self.x) * tf.compat.v1.log(epsilon + 1 - self.x_hat),
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
        self.train_op = tf.compat.v1.train.AdamOptimizer(
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


def vae( train_data, learning_rate=0.000001,
            batch_size=10, num_epoch=700, n_z=16):        # 700
    # vae mape:  35.04757759101026  rmse:  2.516632172941658        50轮
    # 34.13594542330274  rmse:  2.4458262870458998 300
    # Create a model
    input_dim = train_data.shape[-1]
    model = VariantionalAutoencoder(input_dim=input_dim,
        learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)

    # Training loop
    for epoch in range(num_epoch):
        # start_time = time.time()

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

        # end_time = time.time()

        # Log the loss
        # if epoch % log_step == 0:
        #     log_str = '[Epoch {}] '.format(epoch)
        #     for k, v in losses.items():
        #         log_str += '{}: {:.3f}  '.format(k, v)
        #     log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
        #     print(log_str)
    reconstruct_data = model.reconstructor(train_data)

    imputed_data = np.where(train_data!=0,train_data,reconstruct_data)
    # print('Done!')
    return imputed_data