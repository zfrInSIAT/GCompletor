import numpy as np
import pandas as pd 
from .metrics import rmse,mape
from numpy.linalg import inv as inv
import tqdm
def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

def cp_combine(U, V, X):
    return np.einsum('is, js, ts -> ijt', U, V, X)

import numpy as np
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

def TRTF(sparse_tensor_ori, rank=30, time_lags=(1,2,24),
         lambda_u=500, lambda_v=500, lambda_ar=500,
         eta=2e-2, lambda_theta=100, maxiter=1000):
    sparse_tensor = sparse_tensor_ori
    dim1, dim2, dim3 = sparse_tensor.shape

    U = 0.1 * np.random.rand(dim1, rank)
    V = 0.1 * np.random.rand(dim2, rank)
    X = 0.1 * np.random.rand(dim3, rank)

    time_lags=np.array(time_lags)
    theta = 0.1 * np.random.rand(time_lags.shape[0], rank)

    binary_tensor = np.zeros((dim1, dim2, dim3))
    # position = np.where(sparse_tensor > 0)

    position = None
    if np.isnan(sparse_tensor).any() == False:
        position = sparse_tensor != 0

    elif np.isnan(sparse_tensor).any() == True:

        position = ~np.isnan(sparse_tensor)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    
    binary_tensor[position] = 1
    # pos = np.where((dense_tensor > 0) & (sparse_tensor == 0))
    d = len(time_lags)
    rank = U.shape[1]

    for iters in tqdm.trange(maxiter):
        var1 = kr_prod(X, V).T
        var2 = kr_prod(var1, var1)
        var3 = (np.matmul(var2, ten2mat(binary_tensor, 0).T).reshape([rank, rank, dim1])
                + np.dstack([lambda_u * np.eye(rank)] * dim1))
        var4 = np.matmul(var1, ten2mat(sparse_tensor, 0).T)
        for i in range(dim1):
            var_Lambda1 = var3[:, :, i]
            inv_var_Lambda1 = inv((var_Lambda1 + var_Lambda1.T) / 2)
            U[i, :] = np.matmul(inv_var_Lambda1, var4[:, i])

        var1 = kr_prod(X, U).T
        var2 = kr_prod(var1, var1)
        var3 = (np.matmul(var2, ten2mat(binary_tensor, 1).T).reshape([rank, rank, dim2])
                + np.dstack([lambda_v * np.eye(rank)] * dim2))
        var4 = np.matmul(var1, ten2mat(sparse_tensor, 1).T)
        for j in range(dim2):
            var_Lambda1 = var3[:, :, j]
            inv_var_Lambda1 = inv((var_Lambda1 + var_Lambda1.T) / 2)
            V[j, :] = np.matmul(inv_var_Lambda1, var4[:, j])

        var1 = kr_prod(V, U).T
        var2 = kr_prod(var1, var1)
        var3 = np.matmul(var2, ten2mat(binary_tensor, 2).T).reshape([rank, rank, dim3])
        var4 = np.matmul(var1, ten2mat(sparse_tensor, 2).T)
        for t in range(dim3):
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < max(time_lags):
                Pt = np.zeros((rank, rank))
                Qt = np.zeros(rank)
            else:
                Pt = np.eye(rank)
                Qt = np.einsum('ij, ij -> j', theta, X[t - time_lags, :])
            if t < dim3 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim3 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim3)))[0]
                for k in index:
                    theta0 = theta.copy()
                    theta0[k, :] = 0
                    Mt = Mt + np.diag(theta[k, :] ** 2)
                    Nt = Nt + np.multiply(theta[k, :], (X[t + time_lags[k], :]
                                                        - np.einsum('ij, ij -> j', theta0,
                                                                    X[t + time_lags[k] - time_lags, :])))
                X[t, :] = np.matmul(
                    inv(var3[:, :, t] + lambda_ar * Pt + lambda_ar * Mt + lambda_ar * eta * np.eye(rank)),
                    (var4[:, t] + lambda_ar * Qt + lambda_ar * Nt))
            elif t >= dim3 - np.min(time_lags):
                X[t, :] = np.matmul(inv(var3[:, :, t] + lambda_ar * Pt + lambda_ar * eta * np.eye(rank)),
                                    (var4[:, t] + Qt))

        for k in range(d):
            theta0 = theta.copy()
            theta0[k, :] = 0
            mat0 = np.zeros((dim3 - np.max(time_lags), rank))
            for L in range(d):
                mat0 += np.matmul(X[np.max(time_lags) - time_lags[L]: dim3 - time_lags[L], :], np.diag(theta0[L, :]))
            VarPi = X[np.max(time_lags): dim3, :] - mat0
            var1 = np.zeros((rank, rank))
            var2 = np.zeros(rank)
            for t in range(np.max(time_lags), dim3):
                B = X[t - time_lags[k], :]
                var1 += np.diag(np.multiply(B, B))
                var2 += np.matmul(np.diag(B), VarPi[t - np.max(time_lags), :])
            theta[k, :] = np.matmul(inv(var1 + lambda_theta * np.eye(rank) / lambda_ar), var2)

        tensor_hat = cp_combine(U, V, X)
        # mape = np.sum(np.abs(dense_tensor[pos] - tensor_hat[pos]) / dense_tensor[pos]) / dense_tensor[pos].shape[0]
        # rmse = np.sqrt(np.sum((dense_tensor[pos] - tensor_hat[pos]) ** 2) / dense_tensor[pos].shape[0])

        # if (iters + 1) % 200 == 0:
        #     print('Iter: {}'.format(iters + 1))
        #     print('MAPE: {:.6}'.format(mape))
        #     print('RMSE: {:.6}'.format(rmse))
        #     print()

    return tensor_hat

def test_TRTF():
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

    # TRTF(sparse_tensor_ori, rank=30, time_lags=(1, 2, 24),
    #      lambda_u=500, lambda_v=500, lambda_ar=500,
    #      eta=2e-2, lambda_theta=100, maxiter=1000)
    TRTF_res2 = TRTF(sparse_tensor2, rank=50,time_lags=(1,2,288),maxiter=200).reshape(dense_mat.shape)

    TRTF_res2_mape2 = mape(dense_mat[pos2], TRTF_res2[pos2])
    TRTF_res2_rmse2 = rmse(dense_mat[pos2], TRTF_res2[pos2])

    print("TRTF_res2_mape2", TRTF_res2_mape2)
    print("TRTF_res2_rmse2", TRTF_res2_rmse2)


if __name__ == "__main__":
    test_TRTF()

    # TRTF_res2_mape2
    # 7.445343356725806
    # TRTF_res2_rmse2
    # 4.509373684618585