import statsmodels.imputation.bayes_mi as bm
import pandas as pd
from .metrics import rmse,mape
import numpy as np
def BayesGauss( data,mean_prior=None, cov_prior=None, cov_prior_df=1):
    # data中需要补全的值必须时np.nan，不能是0
    
    nan_mat = data.copy()
    model = bm.BayesGaussMI(nan_mat,mean_prior=mean_prior,
                            cov_prior=cov_prior, cov_prior_df=cov_prior_df)
    model.update()
    return nan_mat

def BayesGausstest():
    dense_mat = pd.read_csv('./datasets/Seattle-data-set/mat.csv', index_col=0)
    rm = pd.read_csv('./datasets/Seattle-data-set/RM_mat.csv', index_col=0)
    dense_mat = dense_mat.values
    rm = rm.values

    binary_mat2 = np.round(rm + 0.5 - 0.2)
    nan_mat2 = binary_mat2.copy()

    nan_mat2[nan_mat2 == 0] = np.nan

    sparse_mat2 = np.multiply(nan_mat2, dense_mat)

    pos2 = np.where((dense_mat != 0) & (binary_mat2 == 0))

    # sparse_tensor2 = sparse_mat2.reshape([sparse_mat2.shape[0], 28, 288])

    BayesGauss_res2 = BayesGauss(sparse_mat2)

    BayesGauss_res2_mape2 = mape(dense_mat[pos2], BayesGauss_res2[pos2])
    BayesGauss_res2_rmse2 = rmse(dense_mat[pos2], BayesGauss_res2[pos2])

    print("BayesGauss_res2_mape2", BayesGauss_res2_mape2)
    print("BayesGauss_res2_rmse2", BayesGauss_res2_rmse2)

def again():
    test_data = np.random.random((4,5))
    sparse_data = test_data.copy()
    sparse_data[:,3]=np.nan
    sparse_data = np.transpose(sparse_data,[1,0])
    # sparse_data[1,3]=0.3
    print(sparse_data)
    bn_res = BayesGauss(sparse_data)
    print(bn_res)
    bn_mape = mape(np.transpose(test_data,[1,0]),bn_res)
    print(bn_mape)

if __name__ == "__main__":
    # test_BayesGauss()
    again()
    # BayesGauss_res2_mape2
    # 19.22978707697564
    # BayesGauss_res2_rmse2
    # 9.108397879247361