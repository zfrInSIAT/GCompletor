import numpy as np

true_all_data = np.load('./all_speed_data.npz')
true_all_speed_feature, true_all_mask = true_all_data['arr_0'][10:], true_all_data['arr_1'][10:]