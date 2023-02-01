import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import Layer,Dropout


class GraphAttentionLayer(keras.layers.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 adj,
                 nodes_num,
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

        self.support = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
        # self.support = adj        # 直接用邻接矩阵做输入的话，会oom。

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
        self.conv1d_1 = keras.layers.Conv1D(self.output_dim, 1, use_bias=False)
        self.conv1d_2 = keras.layers.Conv1D(1, 1)
        self.conv1d_3 = keras.layers.Conv1D(1, 1)
        self.build = True
        print('[GAT LAYER]: GAT W & b built.')

    def call(self, inputs, training=True):
        # 完成输入到输出的映射关系
        # inputs = tf.nn.l2_normalize(inputs, 1)

        raw_shape = inputs.shape   # (nodes_num,11)
        outputs = []
        for i in range(self.attn_heads):

            # conv1d 的输入至少是三维，所以需要reshape。
            inputs = tf.reshape(inputs, shape=(1, raw_shape[0], raw_shape[1]))  # (1, nodes_num, input_dim)
            mapped_inputs = self.conv1d_1(inputs)  # (1, nodes_num, output_dim)
            # mapped_inputs = tf.nn.l2_normalize(mapped_inputs)

            sa_1 = self.conv1d_2(mapped_inputs)  # (1, nodes_num, 1)
            sa_2 = self.conv1d_3(mapped_inputs)  # (1, nodes_num, 1)

            con_sa_1 = tf.reshape(sa_1, shape=(raw_shape[0], 1))  # (nodes_num, 1)
            con_sa_2 = tf.reshape(sa_2, shape=(raw_shape[0], 1))  # (nodes_num, 1)

            con_sa_1 = tf.cast(self.support, dtype=tf.float32) * con_sa_1  # (nodes_num, nodes_num) W_hi   (nodes_num,1) #原作者的这个注释就有问题，得到的是（nodes_num，nodes_num)
            con_sa_2 = tf.cast(self.support, dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])  # (nodes_num, nodes_num) W_hj   (nodes_num, nodes_num)

            weights = tf.sparse.add(con_sa_1,con_sa_2)

            # weights = tf.sparse.add(con_sa_1, con_sa_2)  # concatenation
            weights_act = tf.SparseTensor(indices=weights.indices,
                                          values=tf.nn.leaky_relu(weights.values),
                                          dense_shape=weights.dense_shape)  # 注意力互相关系数
            attention = tf.sparse.softmax(weights_act)  # 输出注意力机制
            # weights_act = tf.nn.leaky_relu(weights)
            # attention = tf.nn.softmax(weights_act)

            inputs = tf.reshape(inputs, shape=raw_shape)        # （nodes_num，nodes_num）
            if self.coef_drop > 0.0:
                attention = tf.SparseTensor(indices=attention.indices,
                                            values=tf.nn.dropout(attention.values, self.coef_dropout),
                                            dense_shape=attention.dense_shape)
            # if training and self.dropout_rate > 0.0:
            inputs = tf.nn.dropout(inputs, self.dropout_rate)
            # if not training:
            #     print("[GAT LAYER]: GAT not training now.")

            attention = tf.sparse.reshape(attention, shape=[self.nodes_num, self.nodes_num])
            # attention = tf.reshape(attention, shape=[self.nodes_num, self.nodes_num])

            value = tf.matmul(inputs, self.kernels[i])
            value = tf.sparse.sparse_dense_matmul(attention, value)
            # value = tf.matmul(attention, value)

            if self.use_bias:
                ret = tf.add(value, self.biases[i])
            else:
                ret = tf.reshape(value, (raw_shape[0], self.output_dim))
            outputs.append(ret)
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)  # (N x KF')
        else:
            output = tf.math.reduce_mean(tf.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_signature(self, input_signature):
        pass

class Timeserial_GAT(Layer):
    def __init__(self,output_dim,adj ,gat_heads,**kwargs):
        self.output_dim = output_dim
        self.adj=adj
        self.heads = gat_heads
        super(Timeserial_GAT,self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape)>=2
        self.feature_input = input_shape[-1]
        self.node_num = input_shape[-2]
        self.gat_1 = GraphAttentionLayer(self.feature_input, self.output_dim, self.adj, self.node_num,attn_heads=self.heads)
        self.drop = Dropout(0.1)
        # self.gat_2=GraphAttentionLayer(self.inpshp, self.output_dim,self.adj, self.node_num)
        self.build=True
    def call(self, inputs, *args, **kwargs):
        # X_in, A_in = inputs
        X_in = inputs

        # l = []
        l2 = []

        for i in range(X_in.shape[0]):
            l = []
            for j in range(X_in.shape[1]):
                gat = self.gat_1(X_in[i, j])
                dp = self.drop(gat)
                # gat2 = self.gat_2(dp)
                l.append(dp)
            sta = tf.stack(l, 0)  # 从axis=1处堆起来，得到的维度为（None,288，4248，F_）
            l2.append(sta)
        return tf.stack(l2,0)       # 还是输出这样的维度吧，不然不容易理解这个模块做了什么，输出shape为(批，288，4248，F_).
    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[1],input_shape[2],self.output_dim # input_shape[:3]+(self.output_dim,)