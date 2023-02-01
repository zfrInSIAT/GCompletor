import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

seed = 22


def my_loss(y_true, y_pred):
    mask = y_true != 0
    return tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_true[mask], y_pred[mask])), axis=-1)


def prepare_data():
    data_list = list(map(np.math.sin, map(lambda x: 0.1 * x, range(1, 200))))
    return np.array([data_list[i:i + 32] for i in range(200 - 32)])


def train():
    ori_data = np.expand_dims(prepare_data(), axis=-1)
    ori_data_shape = ori_data.shape
    print(ori_data_shape)
    model_input = keras.layers.Input(shape=(32,1))
    # model_input2 = tf.expand_dims(model_input, axis=-1)
    gru = keras.layers.GRU(units=16, return_sequences=True, go_backwards=True)
    bigru_output = keras.layers.Bidirectional(layer=gru, merge_mode='sum')(model_input)
    dense_output = keras.layers.Dense(units=1)(bigru_output)
    # leaky_relu = tf.nn.leaky_relu(dense_output)
    # dense_output2 = keras.layers.Dense(units=1)(leaky_relu)
    # dense_output3 = tf.squeeze(dense_output2)
    model = keras.Model(inputs=model_input, outputs=dense_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.007),
                  loss=my_loss,
                  )

    np.random.seed(seed)
    train_data = np.where(np.random.random(ori_data_shape) < 0.2, 0., ori_data)
    model.fit(train_data,
              train_data,
              batch_size=20,
              epochs=10,
              shuffle=True,
              verbose=0)
    pret = model.predict(train_data)
    print(np.squeeze(train_data[0]))
    print(np.squeeze(pret[0]))
    print(np.squeeze(ori_data[0]))


if __name__ == "__main__":
    train()
