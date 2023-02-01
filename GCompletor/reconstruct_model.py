import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from gat_layer3 import  GraphAttentionLayer
adj = np.load('./sparse_adj.npy', allow_pickle=True)
inputs = tf.random.normal(shape=(3,4248,13))
output_dim = 3
# inputs = tf.constant([2,3,4,5,6,7,8,9],shape=(2,2,2),dtype='float32')
support = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
raw_shape = inputs.shape
conv1d_2 = keras.layers.Conv1D(1, 1)
conv1d_1 = keras.layers.Conv1D(output_dim, 1, use_bias=False)
mapped_inputs = conv1d_1(inputs)
sa_1 = conv1d_2(mapped_inputs)
con_sa_1 = tf.reshape(sa_1, shape=(-1,raw_shape[-2], 1))
con_sa_1 = tf.cast(support, dtype=tf.float32) * con_sa_1
my_test_layer = GraphAttentionLayer(input_dim=13,output_dim=3,adj=adj,nodes_num=4248)
outputs = my_test_layer()
print(outputs.shape)
