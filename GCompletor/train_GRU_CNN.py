import numpy as np
import tensorflow as tf
from model import GRU_CNN
import sys
sys.path.append(r'./baselines')
from baselines.metrics import mape,rmse



true_feature, true_mask, train_feature, train_mask, speed_feature, train_speed_feature = np.load(
    './feature.npz').values()

mask = true_mask ^ train_mask
model = GRU_CNN()
mc_callback = tf.keras.callbacks.ModelCheckpoint('./logs/best_model4GRU_CNN.h5',
                              monitor='val_acc',
                              save_best_only=True,
                              save_weights_only=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              loss='mean_absolute_error',
              weighted_metrics=['acc'],
              )
# model.build(tf.transpose(train_feature,[1,0,2]).shape)
# model.summary()
model.fit(tf.transpose(train_feature,[1,0,2]),
          tf.transpose(train_speed_feature,[1,0,2]),
          validation_data=[tf.transpose(true_feature,[1,0,2]),tf.transpose(speed_feature,[1,0,2])],
          batch_size=10,
          epochs=30,
          callbacks=[mc_callback])

model.load_weights('./logs/best_model4GRU_CNN.h5')
pre = model.predict(tf.transpose(train_feature,[1,0,2]),batch_size=None)

y_true = speed_feature[mask]
y_pre = tf.transpose(pre,[1,0,2])[mask]

model_mape = mape(y_true,y_pre)
model_rmse = rmse(y_true,y_pre)

print("GRU_CNN mape:{}\trmse:{}".format(model_mape,model_rmse))
# GRU_CNN mape:90.39978269949039	rmse:10.156549228855724
