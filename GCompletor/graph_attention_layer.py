'''
https://github.com/danielegrattarola/keras-gat

'''
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 adj,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super(GraphAttention, self).__init__()
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False
        self.support = tf.cast(tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2]), 'float32')
        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

    def build(self, input_shape):
        # print(input_shape)
        F = input_shape[-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head), )
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs, training=True):
        raw_shape = inputs.shape
        # X = tf.squeeze(inputs)
        X=inputs
        # X = tf.reshape(inputs,shape=(-1,inputs.shape[-2],inputs.shape[-1]))  # Node features (N x F)  修改过了
        # A = inputs[1]  # Adjacency matrix (N x N)
        # print(X.shape,A.shape)
        outputs = []

        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])  # (N x 1), [a_1]^T [Wh_i]   # 矩阵乘法
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            # dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting
            dense = attn_for_self + K.transpose(attn_for_neighs)

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            # 这个得去掉，因为当A是稀疏矩阵时，这样计算会出错
            # mask = -10e9 * (1.0 - A)
            # dense += mask
            dense = self.support * dense
            # 修改处
            # dense = tf.sparse.SparseTensor(indices=self.support.indices,
            #                                values=tf.sparse.sparse_dense_matmul(self.support, dense),
            #                                dense_shape=self.support.dense_shape)
            # dense = self.support * dense
            dropout_attn = tf.sparse.softmax(dense)
            # Apply softmax to get attention coefficients
            # dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            # dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            # node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')
            # print(type(dropout_attn),dropout_attn)
            # print(type(dropout_feat), dropout_feat)
            node_features = tf.sparse.sparse_dense_matmul(dropout_attn, dropout_feat)
            # node_features = dropout_feat

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            # node_features = tf.expand_dims(node_features,axis=0)
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # 这里修改过了，之前是input_shape[0][0]，因为之前作者将图的节点数视作批的数量，所以之前的输出是N*F_(单头或多头average时）。
        # 改成input_shape[0][2]对应[(batch_size,时间，道路数，特征数),(道路数，道路数)]中的道路数量。
        # output_shape = input_shape[:-1], self.output_dim
        # return output_shape
        pass

# class Timeserial_GAT(Layer):
#     def __init__(self, output_dim, adj, gat_heads=2, **kwargs):
#         self.output_dim = output_dim
#         self.adj = adj
#         self.gat_heads = gat_heads
#         super(Timeserial_GAT, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         self.gat_1 = GraphAttention(F_=self.output_dim, adj=self.adj, attn_heads=self.gat_heads)
#         # self.drop = Dropout(0.1)
#         # self.gat_2 = GraphAttention(F_=self.output_dim, adj=self.adj, attn_heads=self.gat_heads)
#         self.timedistributed = TimeDistributed(self.gat_1, input_shape=(input_shape[-2], input_shape[-1]))
#         self.build = True
#
#     def call(self, inputs, *args, **kwargs):
#         # X_in, A_in = inputs
#         # # l = []
#         # l2 = []
#         output = self.timedistributed(inputs)
#         # for i in range(X_in.shape[0]):
#         #     l = []
#         #     for j in range(X_in.shape[1]):
#         #         gat = self.gat_1([X_in[i, j], A_in[0]])
#         #         dp = self.drop(gat)
#         #         gat2 = self.gat_2([dp, A_in[0]])
#         #         l.append(gat2)
#         #     sta = tf.stack(l, 0)  # 从axis=1处堆起来，得到的维度为（None,288，4248，F_）
#         #     l2.append(sta)
#         # return tf.stack(l2, 0)  # 还是输出这样的维度吧，不然不容易理解这个模块做了什么，输出shape为(批，288，4248，F_).
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], input_shape[2], self.output_dim  # input_shape[:3]+(self.output_dim,)
