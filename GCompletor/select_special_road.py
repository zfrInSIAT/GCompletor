import numpy as np

selected500road_list_and_adj_mat = np.load("./data/selected500road_index_and_adj_mat.npz")
roadsectId_list = selected500road_list_and_adj_mat['arr_0']  # 被选中的道路在原先4248条道路中的编号
roadsect_adj_mat = selected500road_list_and_adj_mat['arr_1']  # 被选中的道路的邻接矩阵
# 数据加载与处理
true_all_data = np.load('./all_speed_data.npz')
true_all_speed_feature, true_all_mask = true_all_data['arr_0'], true_all_data['arr_1']
# 保存的shape为（14，288，4248,1），需要squeeze一下
true_all_speed_feature, true_all_mask = np.squeeze(true_all_speed_feature[:, :, roadsectId_list, :]), np.squeeze(true_all_mask[:, :, roadsectId_list, :])
true_all_speed_feature = true_all_speed_feature.transpose([2, 0, 1])
true_all_mask = true_all_mask.transpose([2, 0, 1])
true_all_speed_feature = np.reshape(true_all_speed_feature, [500, -1])
true_all_mask = np.reshape(true_all_mask,[500, -1])
road_list = []
for i in range(500):
    tmp = true_all_speed_feature[i][true_all_mask[i]]
    var_res = np.var(tmp)
    road_list.append((var_res,i))
_,sorted_list = zip(*sorted(road_list, reverse=False,key=lambda x:x[0]))
sorted_list = list(sorted_list)

np.save(r'./data/sorted_var_500_road', sorted_list)