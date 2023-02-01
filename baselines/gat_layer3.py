'''
该文件是对get_layer2的修改，区别在于，对输入的维度进行了调整，可适应多个维度。
tensorflow version==2.3, 2.3以上的版本已经支持conv1d输入多个维度。
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import Layer, Dropout


class GraphAttentionLayer(Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 adj,
                 nodes_num,
                 feature_of_edge=None,
                 attn_heads=1,
                 attn_heads_reduction='average',  # {'concat', 'average'}
                 dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 coef_dropout=0.0,
                 **kwargs):
        """
        :param input_dim: 输入的维度
        :param output_dim: 输出的维度，不等于input_dim
        :param adj: 具有自环的tuple类型的邻接表[coords, values, shape]， 可以采用sp.coo_matrix生成
        :param nodes_num: 点数量
        :param dropout_rate: 丢弃率，防过拟合，默认0.5
        :param activation: 激活函数
        :param use_bias: 偏移，默认True
        :param kernel_initializer: 权值初始化方法
        :param bias_initializer: 偏移初始化方法
        :param kernel_regularizer: 权值正则化
        :param bias_regularizer: 偏移正则化
        :param activity_regularizer: 输出正则化
        :param kernel_constraint: 权值约束
        :param bias_constraint: 偏移约束
        :param coef_dropout: 互相关系数丢弃，默认0.0
        :param kwargs:
        """
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        super(GraphAttentionLayer, self).__init__()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_dim = input_dim
        # self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.coef_drop = coef_dropout

        # self.support = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
        self.support = adj        # 直接用邻接矩阵做输入的话，会oom。
        self.feature_of_edge = feature_of_edge
        self.nodes_num = nodes_num
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        # self.kernel = None
        self.mapping = None
        # self.bias = None

        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = output_dim * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = output_dim

    def build(self, input_shape):
        """
        只执行一次
        虽然使用了多头机制，但是默认设置多头的输出维度是相同的
        """
        for i in range(self.attn_heads):
            kernel = self.add_weight(shape=(self.input_dim, self.output_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     trainable=True,
                                     name='gat_kernel{}'.format(i))
            self.kernels.append(kernel)
            if self.use_bias:
                bias = self.add_weight(shape=(self.nodes_num, self.output_dim),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint,
                                       trainable=True,
                                       name='gat_bias{}'.format(i))
                self.biases.append(bias)
        # self.conv1d_1 = keras.layers.Conv1D(self.output_dim, 1, use_bias=False)
        # self.conv1d_2 = keras.layers.Conv1D(1, 1)
        # self.conv1d_3 = keras.layers.Conv1D(1, 1)
            conv1d_1 = keras.layers.Conv1D(self.output_dim, 1, use_bias=False)
            conv1d_2 = keras.layers.Conv1D(1, 1)
            conv1d_3 = keras.layers.Conv1D(1, 1)
            if self.feature_of_edge is not None:
                edge_weight = self.add_weight(shape=(self.nodes_num, self.nodes_num),
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint,
                                              trainable=True,
                                              name='gat_edge_weight{}'.format(i))
            else:
                edge_weight = None
            self.attn_kernels.append([conv1d_1,conv1d_2,conv1d_3,edge_weight])
        self.build = True
        # print('[GAT LAYER]: GAT W & b built.')

    def call(self, inputs, training=True):
        # 完成输入到输出的映射关系
        # inputs = tf.nn.l2_normalize(inputs, 1)

        raw_shape = inputs.shape  # (nodes_num,11)
        outputs = []
        for i in range(self.attn_heads):

            # 因为tensorflow2.1的版本中conv1d只支持shape rank为3的输入，所以必须要reshape为三维的张量
            # 2.3以上的版本支持conv1d输入多个维度，但只对最后两个维度进行运算。
            # inputs = tf.reshape(inputs, shape=(-1, raw_shape[-2], raw_shape[-1]))  # (1, nodes_num, input_dim)
            mapped_inputs = self.attn_kernels[i][0](inputs)  # (1, nodes_num, output_dim)  (None,288,4248,f_)
            # mapped_inputs = tf.nn.l2_normalize(mapped_inputs)

            sa_1 = self.attn_kernels[i][1](mapped_inputs)  # (1, nodes_num, 1)        (None,288,4248,1)
            sa_2 = self.attn_kernels[i][2](mapped_inputs)  # (1, nodes_num, 1)

            # con_sa_1 = tf.reshape(sa_1, shape=(raw_shape[:-1] + [1]))  # (nodes_num, 1)
            # con_sa_2 = tf.reshape(sa_2, shape=(raw_shape[:-1] + [1]))  # (nodes_num, 1)

            # support要这样转换，也是因为sparse tensor进行点乘，必须要维度一致,但是tensor不需要维度一致，还能进行广播。
            # con_sa_1 = tf.sparse.from_dense(
            #     tf.sparse.to_dense(tf.cast(self.support, dtype=tf.float32)) * sa_1)  # (None,288,4248,4248)
            # con_sa_2 = tf.sparse.from_dense(
            #     tf.sparse.to_dense(tf.cast(self.support, dtype=tf.float32)) * tf.transpose(sa_2, list(
            #         range(sa_2.shape.rank - 2)) + [sa_2.shape.rank - 1, sa_2.shape.rank - 2]))
            con_sa_1 = tf.cast(self.support, dtype=tf.float32) * sa_1  # (None,288,4248,4248)
            con_sa_2 = tf.cast(self.support, dtype=tf.float32) * tf.transpose(sa_2, list(
                    range(sa_2.shape.rank - 2)) + [sa_2.shape.rank - 1, sa_2.shape.rank - 2])
            # print("con_sa_1.shape", con_sa_1.shape)
            # print("con_sa_2.shape", con_sa_2.shape)
            # weights = tf.sparse.add(con_sa_1, con_sa_2)
            weights = tf.add(con_sa_1, con_sa_2)
            if self.feature_of_edge is not None:
                edge_wei = tf.multiply(self.attn_kernels[i][3], self.feature_of_edge)

                weights_contain_edge_feature = tf.add(weights, edge_wei)
            else:
                weights_contain_edge_feature = weights
                # print("weights.shape", weights.shape)
            weights_act = tf.keras.layers.LeakyReLU()(weights_contain_edge_feature)
            attention = tf.nn.softmax(weights_act)  # 输出注意力机制       (None,288,4248,4248)

            inputs = tf.nn.dropout(inputs, self.dropout_rate)
            value = tf.matmul(inputs, self.kernels[i])  # (None,288,4248,f_)
            # print("value.shape", value.shape)
            value = tf.matmul(attention, value)
            # print("gat,value.shape:", value.shape)
            if self.use_bias:
                ret = tf.add(value, self.biases[i])
            else:
                ret = value
            outputs.append(ret)
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)  # (N x KF')
        else:
            output = tf.math.reduce_mean(tf.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        # print("output.shape", output.shape)
        return output

    def compute_output_signature(self, input_signature):
        pass

class TimeGAT(Layer):
    def __init__(self, input_dim, output_dim, adj, nodes_num):
        super(TimeGAT, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.nodes_num = nodes_num

    def build(self, input_shape):
        assert len(input_shape)==4
        self.gat_list = []
        self.time_slot = input_shape[1]
        for i in range(self.time_slot):
            self.gat_list.append(GraphAttentionLayer(self.input_dim, self.output_dim, self.adj, self.nodes_num))

    def call(self, inputs, **kwargs):
        tmp_list = []
        for i in range(self.time_slot):
            tmp_list.append(self.gat_list[i](inputs[:, i]))
        output = tf.stack(tmp_list, axis=1)
        return output
