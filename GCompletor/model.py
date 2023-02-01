import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Conv1D, Concatenate
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, Model
from graph_attention_layer import GraphAttention


# from tensorflow.keras.losses import


class MyModel(tf.keras.Model):
    def __init__(self,F_=7,n_attn_heads=1,dropout_rate=0.4,l2_reg=5e-4/2 ,gru_output_dim=5,
                 conv1d_filters=5,conv1d_kernel_size=5,conv1d_strides=1,conv1d_padding='same'):
        super(MyModel, self).__init__()
        self.graph_attention_1 = GraphAttention(F_,
                                                attn_heads=n_attn_heads,
                                                attn_heads_reduction='concat',
                                                dropout_rate=dropout_rate,
                                                activation='elu',
                                                kernel_regularizer=l2(l2_reg),
                                                attn_kernel_regularizer=l2(l2_reg))
        self.graph_attention_2 = GraphAttention(F_,
                                                attn_heads=n_attn_heads,
                                                attn_heads_reduction='concat',
                                                dropout_rate=dropout_rate,
                                                activation='elu',
                                                kernel_regularizer=l2(l2_reg),
                                                attn_kernel_regularizer=l2(l2_reg))
        self.forward_layer = GRU(gru_output_dim, return_sequences=True)
        self.backward_layer = GRU(gru_output_dim, return_sequences=True, go_backwards=True)
        self.bigru = Bidirectional(self.forward_layer, backward_layer=self.backward_layer, merge_mode='concat')

        self.conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                             strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]

        self.fconcat = Concatenate()  # ([x, y])

        # decoder
        self.forward_layer2 = GRU(1, return_sequences=True)  # 这里好像没法指定输出维度和状态维度不相同
        self.backward_layer2 = GRU(1, return_sequences=True, go_backwards=True)
        self.decoder_bigru = Bidirectional(self.forward_layer2, backward_layer=self.backward_layer2, merge_mode='ave')

    def call(self, inputs, training=None, mask=None):
        X_in, A_in = inputs
        l=[]
        for i in range(X_in.shape[0]):
            gat = self.graph_attention_1([X_in[i],A_in])
            l.append(gat)
        sta=tf.stack(l,1)
        output=[]
        for i in range(sta.shape[0]):
            bg=self.bigru(sta[i])
            con=self.conv1d(sta[i])
            cc=self.fconcat([bg,con])
            dec=self.decoder_bigru(cc)
            output.append(dec)
        ret=tf.stack(output,1)
        return ret


def mymodel(A_in):
    learning_rate = 5e-3  # Learning rate for Adam
    epochs = 10000

    es_patience = 100
    F_ = 7
    n_attn_heads = 1
    dropout_rate = 0.4
    l2_reg = 5e-4 / 2
    gru_output_dim = 5
    conv1d_filters = 5
    conv1d_kernel_size = 5
    conv1d_strides = 1
    conv1d_padding = 'valid'
    graph_attention_1 = GraphAttention(F_,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='concat',
                                       dropout_rate=dropout_rate,
                                       activation='elu',
                                       kernel_regularizer=l2(l2_reg),
                                       attn_kernel_regularizer=l2(l2_reg))
    graph_attention_2 = GraphAttention(F_,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='concat',
                                       dropout_rate=dropout_rate,
                                       activation='elu',
                                       kernel_regularizer=l2(l2_reg),
                                       attn_kernel_regularizer=l2(l2_reg))
    forward_layer = GRU(gru_output_dim, return_sequences=True)
    backward_layer = GRU(gru_output_dim, return_sequences=True, go_backwards=True)
    bigru = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='concat')

    conv1d = Conv1D(filters=conv1d_filters, kernel_size=conv1d_kernel_size,
                    strides=conv1d_strides, padding=conv1d_padding)  # input_shape=input_shape[2:]

    fconcat = Concatenate  # ([x, y])

    # decoder
    forward_layer2 = GRU(1, return_sequences=True)  # 这里好像没法指定输出维度和状态维度不相同
    backward_layer2 = GRU(1, return_sequences=True, go_backwards=True)
    decoder_bigru = Bidirectional(forward_layer2, backward_layer=backward_layer2, merge_mode='ave')

    X_in = Input(shape=(288, 4248, 11))
    # A_in = Input(shape=(5, 5))
    l = []
    for i in range(288):
        gat = graph_attention_1([X_in[i], A_in])
        gat2= graph_attention_2(gat)
        l.append(gat2)
    sta = tf.stack(l, 1)
    output = []
    for i in range(sta.shape[0]):
        bg = bigru(sta[i])
        con = conv1d(sta[i])
        cc = fconcat([bg, con])
        dec = decoder_bigru(cc)
        output.append(dec)
    ret = tf.stack(output, 1)

    model = Model(inputs=[X_in, A_in], outputs=ret)
    return model

# 直接不要图注意力试试

class GRU_CNN(Model):
    def __init__(self,gru_output_dim=16,cnn_output_dim=16,cnn_kernel_size =5,cnn_strides=1,cnn_padding='same',**kwargs):
        super(GRU_CNN, self).__init__(**kwargs)
        self.gru_output_dim = gru_output_dim
        self.cnn_output_dim = cnn_output_dim
        self.cnn_kernel_size=cnn_kernel_size
        self.cnn_strides=cnn_strides
        self.cnn_padding=cnn_padding

    def build(self, input_shape):
        # self.dropout = Dropout(0.05)
        # self.dropout2 = Dropout(0.01)

        self.forward_layer = GRU(self.gru_output_dim, return_sequences=True)
        self.backward_layer = GRU(self.gru_output_dim, return_sequences=True, go_backwards=True)
        self.bigru = Bidirectional(self.forward_layer, backward_layer=self.backward_layer, merge_mode='concat')

        self.conv1d = Conv1D(filters=self.cnn_output_dim, kernel_size=self.cnn_kernel_size,
                             strides=self.cnn_strides, padding=self.cnn_padding)  # input_shape=input_shape[2:]
        self.conv1d2 = Conv1D(filters=self.cnn_output_dim, kernel_size=self.cnn_kernel_size,
                             strides=self.cnn_strides, padding=self.cnn_padding)
        self.dense = Dense(1)
        self.concat = Concatenate()  # ([x, y])

        # decoder
        self.forward_layer2 = GRU(32, return_sequences=True)  # 这里好像没法指定输出维度和状态维度不相同
        self.backward_layer2 = GRU(32, return_sequences=True, go_backwards=True)
        self.decoder_bigru = Bidirectional(self.forward_layer2, backward_layer=self.backward_layer2, merge_mode='concat')
        self.built=True
    def call(self, inputs, training=None, mask=None):
        # dp = self.dropout(inputs)
        bg = self.bigru(inputs)
        cn = self.conv1d(inputs)
        # dp2 = self.dropout2(cn)
        cn2 = self.conv1d2(cn)
        cc = self.concat([bg, cn2])

        dbg = self.decoder_bigru(cc)
        den = self.dense(dbg)
        return den
    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[1],1