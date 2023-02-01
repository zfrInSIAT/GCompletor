from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import pandas as pd
import tqdm
from metrics import mape,rmse
def get_deviation(true_data, mask, pret_data):
    true_data_mean = np.mean(true_data)
    true_data_num = true_data.size
    true_data_distribution_ret = []
    metrics = [-9.75, -9.25, -8.75, -8.25, -7.75, -7.25, -6.75, -6.25, -5.75, -5.25, -4.75, -4.25, -3.75, -3.25, -2.75,
               -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25,
               5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75]
    # true_data_distribution_ret.append(np.sum(true_data[mask]-true_data_mean<-10)/true_data_num)
    for i in range(len(metrics) - 1):
        t = (np.sum(true_data - true_data_mean < metrics[i + 1]) - np.sum(
            true_data - true_data_mean < metrics[i])) / true_data_num
        true_data_distribution_ret.append(t)
    # true_data_distribution_ret.append(np.sum(true_data[mask]-true_data_mean>10)/true_data_num)

    pret_mask = mask == False
    pret_data_mean = np.mean(pret_data[pret_mask])
    pret_data_num = np.sum(pret_mask)
    pret_data_distribution_ret = []
    # pret_data_distribution_ret.append(np.sum(pret_data[pret_mask]<-10)/pret_data_num)
    for i in range(len(metrics) - 1):
        t = (np.sum(pret_data[pret_mask] - pret_data_mean < metrics[i + 1]) - np.sum(
            pret_data[pret_mask] - pret_data_mean < metrics[i])) / pret_data_num
        pret_data_distribution_ret.append(t)
    # pret_data_distribution_ret.append(np.sum(pret_data[pret_mask]>10)/pret_data_num)
    return true_data_distribution_ret, pret_data_distribution_ret
# data=np.array([[0,2,3,4,5,6,0],[3,0,5,6,7,8,0]],dtype=np.float).transpose()
# data=np.array([1,2,5,1,2,5,1,2,5,1,2,5],dtype=np.float)

# mod=ARIMA(data,order=(1,0,0)).fit()
# # res=mod.fit()
# print(mod.predict(5,typ='levels'))
# endog[2]
# endog[1,6]=np.nan

import statsmodels.api as sm
# import numpy as np
# mod = []
d=30
file_path=r'../data/seattle/no_random_missing_time/seattle_no_random_missing_time20.npy'
# file_path=r'../data/seattle/no_random_missing_time/seattle_no_random_missing_time40.npy'
# file_path=r'../data/seattle/no_random_missing_time/seattle_no_random_missing_time60.npy'
# file_path=r'../data/seattle/no_random_missing_time/seattle_no_random_missing_time80.npy'
#
# file_path=r'../data/seattle/no_random_missing_road/seattle_no_random_missing_road20.npy'
# file_path=r'../data/seattle/no_random_missing_road/seattle_no_random_missing_road40.npy'
# file_path=r'../data/seattle/no_random_missing_road/seattle_no_random_missing_road60.npy'
# file_path=r'../data/seattle/no_random_missing_road/seattle_no_random_missing_road80.npy'
#
# file_path=r'../data/seattle/random_missing/seattle_random_missing20.npy'
# file_path=r'../data/seattle/random_missing/seattle_random_missing40.npy'
# file_path=r'../data/seattle/random_missing/seattle_random_missing60.npy'
# file_path=r'../data/seattle/random_missing/seattle_random_missing80.npy'
true_data = np.array(pd.read_csv('../data/seattle/Speed2.csv').values).reshape([288,365,323])
true_data = np.transpose(true_data,[1,0,2])
# 行是时间，列是路段
data = np.transpose(np.load(file_path).reshape([288,365,323]),[1,0,2])
mask = data!=0
mask2 = data==0
# print(mask)
for k in range(data.shape[0]):
    for i in range(data.shape[2]):
        avg = np.mean(data[k,data[k,:,i] != 0])
        for j in range(data.shape[1]):
            if j <= d and data[k,j,i]==0:
                data[k,j,i] = avg

            else:
    #
                if data[k,j, i] == 0:
                    mod = ARIMA(data[k,j-d:j, i],order=(1,0,0)).fit()
                    # mod=mod.fit()
                    data[k,j, i] = mod.predict(1 ,dynamic=True)[0]
true_data_distribution,pret_data_distribution=get_deviation(true_data,mask,data)
print(true_data_distribution)
print(pret_data_distribution)
mean_res2_mape2 = mape(true_data[mask2], data[mask2])
mean_res2_rmse2 = rmse(true_data[mask2], data[mask2])
ret_str = "mean_missing_rate mape={}\n".format(mean_res2_mape2)
ret_str += "mean_missing_rate rmse={}".format(mean_res2_rmse2)
print(ret_str)