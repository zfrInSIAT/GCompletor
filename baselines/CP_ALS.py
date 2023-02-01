import numpy as np
import tqdm
from numpy.linalg import inv as inv
import pandas as pd
from .metrics import rmse,mape

def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

def cp_combine(U, V, X):
    return np.einsum('is, js, ts -> ijt', U, V, X)

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

def CP_ALS( sparse_tensor_ori, rank=50, maxiter=1000):
    

    sparse_tensor = sparse_tensor_ori.copy()
    dim1, dim2, dim3 = sparse_tensor.shape
    dim = np.array([dim1, dim2, dim3])

    U = 0.1 * np.random.rand(dim1, rank)
    V = 0.1 * np.random.rand(dim2, rank)
    X = 0.1 * np.random.rand(dim3, rank)

    if np.isnan(sparse_tensor).any():
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    pos = np.where(sparse_tensor != 0)
    # pos = np.where(~np.isnan(sparse_tensor))

    binary_tensor = np.zeros((dim1, dim2, dim3))
    binary_tensor[pos] = 1
    tensor_hat = np.zeros((dim1, dim2, dim3))

    for iters in tqdm.trange(maxiter):
        for order in range(dim.shape[0]):
            if order == 0:
                var1 = kr_prod(X, V).T
            elif order == 1:
                var1 = kr_prod(X, U).T
            else:
                var1 = kr_prod(V, U).T
            var2 = kr_prod(var1, var1)
            var3 = np.matmul(var2, ten2mat(binary_tensor, order).T).reshape([rank, rank, dim[order]])
            var4 = np.matmul(var1, ten2mat(sparse_tensor, order).T)
            for i in range(dim[order]):
                var_Lambda = var3[:, :, i]
                inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2 + 10e-12 * np.eye(rank))
                vec = np.matmul(inv_var_Lambda, var4[:, i])
                if order == 0:
                    U[i, :] = vec.copy()
                elif order == 1:
                    V[i, :] = vec.copy()
                else:
                    X[i, :] = vec.copy()

        tensor_hat = cp_combine(U, V, X)
    return tensor_hat

def test_CP_ALS():
    dense_mat = pd.read_csv('./datasets/Seattle-data-set/mat.csv', index_col=0)
    rm = pd.read_csv('./datasets/Seattle-data-set/RM_mat.csv', index_col=0)
    dense_mat = dense_mat.values
    rm = rm.values

    binary_mat2 = np.round(rm + 0.5 - 0.2)
    nan_mat2 = binary_mat2.copy()

    nan_mat2[nan_mat2 == 0] = np.nan

    sparse_mat2 = np.multiply(nan_mat2, dense_mat)

    pos2 = np.where((dense_mat != 0) & (binary_mat2 == 0))

    sparse_tensor2 = sparse_mat2.reshape([sparse_mat2.shape[0], 28, 288])
    # sparse_tensor_ori, rank = 30, time_lags = (1, 2, 24),
    # burn_iter = 1100, gibbs_iter = 100
    CP_ALS_res2 = CP_ALS(sparse_tensor2, 50,100).reshape(
        dense_mat.shape)

    CP_ALS_res2_mape2 = mape(dense_mat[pos2], CP_ALS_res2[pos2])
    CP_ALS_res2_rmse2 = rmse(dense_mat[pos2], CP_ALS_res2[pos2])

    print("CP_ALS_res2_mape2", CP_ALS_res2_mape2)
    print("CP_ALS_res2_rmse2", CP_ALS_res2_rmse2)


if __name__ == "__main__":
    test_CP_ALS()

    # CP_ALS_res2_mape2
    # 7.606663196381899
    # CP_ALS_res2_rmse2
    # 4.58422418409836