import numpy as np
import pandas as pd
import tqdm
from numpy.linalg import inv as inv

from .metrics import rmse, mape




def TRMF( sparse_mat_ori, rank = 50, lambda_w=500,
         lambda_x=500,
         lambda_theta=500,
         eta=0.03, time_lags=(1, 2, 144), maxiter=200):
    """Temporal Regularized Matrix Factorization, TRMF."""

    ## Initialize parameters
    sparse_mat=sparse_mat_ori.copy()

    time_lags = np.array(time_lags)
    d = time_lags.shape[0]
    dim1, dim2 = sparse_mat.shape
    W = 0.1 * np.random.rand(dim1, rank)
    X = 0.1 * np.random.rand(dim2, rank)
    theta = 0.1 * np.random.rand(d, rank)

    ## Set hyperparameters


    dim1, dim2 = sparse_mat.shape
    pos_train=None
    if np.isnan(sparse_mat).any() == False:
        pos_train = sparse_mat != 0

    elif np.isnan(sparse_mat).any() == True:

        pos_train = ~np.isnan(sparse_mat)
        sparse_mat[np.isnan(sparse_mat)] = 0

    # pos_train = np.where(sparse_mat != 0)

    binary_mat = sparse_mat.copy()
    binary_mat[pos_train] = 1
    d, rank = theta.shape
    mat_hat=None
    for it in tqdm.trange(maxiter):
        ## Update spatial matrix W
        for i in range(dim1):
            pos0 = np.where(sparse_mat[i, :] != 0)
            Xt = X[pos0[0], :]
            vec0 = Xt.T @ sparse_mat[i, pos0[0]]
            mat0 = inv(Xt.T @ Xt + lambda_w * np.eye(rank))
            W[i, :] = mat0 @ vec0
        ## Update temporal matrix X
        for t in range(dim2):
            pos0 = np.where(sparse_mat[:, t] != 0)
            Wt = W[pos0[0], :]
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < np.max(time_lags):
                Pt = np.zeros((rank, rank))
                Qt = np.zeros(rank)
            else:
                Pt = np.eye(rank)
                Qt = np.einsum('ij, ij -> j', theta, X[t - time_lags, :])
            if t < dim2 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2)))[0]
                for k in index:
                    Ak = theta[k, :]
                    Mt += np.diag(Ak ** 2)
                    theta0 = theta.copy()
                    theta0[k, :] = 0
                    Nt += np.multiply(Ak, X[t + time_lags[k], :]
                                      - np.einsum('ij, ij -> j', theta0, X[t + time_lags[k] - time_lags, :]))
            vec0 = Wt.T @ sparse_mat[pos0[0], t] + lambda_x * Nt + lambda_x * Qt
            mat0 = inv(Wt.T @ Wt + lambda_x * Mt + lambda_x * Pt + lambda_x * eta * np.eye(rank))
            X[t, :] = mat0 @ vec0
        ## Update AR coefficients theta
        for k in range(d):
            theta0 = theta.copy()
            theta0[k, :] = 0
            mat0 = np.zeros((dim2 - np.max(time_lags), rank))
            for L in range(d):
                mat0 += X[np.max(time_lags) - time_lags[L]: dim2 - time_lags[L], :] @ np.diag(theta0[L, :])
            VarPi = X[np.max(time_lags): dim2, :] - mat0
            var1 = np.zeros((rank, rank))
            var2 = np.zeros(rank)
            for t in range(np.max(time_lags), dim2):
                B = X[t - time_lags[k], :]
                var1 += np.diag(np.multiply(B, B))
                var2 += np.diag(B) @ VarPi[t - np.max(time_lags), :]
            theta[k, :] = inv(var1 + lambda_theta * np.eye(rank) / lambda_x) @ var2

        mat_hat = W @ X.T
    return mat_hat


def test_TRMF():
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
    # def TRMF(sparse_mat, lambda_w=500,
    #          lambda_x=500,
    #          lambda_theta=500,
    #          eta=0.03, time_lags=(1, 2, 144), maxiter=200)

    TRMF_res2 = TRMF(sparse_mat2, lambda_w=500,
                     lambda_x=500,
                     lambda_theta=500,
                     eta=0.03, time_lags=(1, 2,3,4, 144), maxiter=200)
    # print(TRMF_res2)
    # print(dense_mat)
    TRMF_res2_mape2 = mape(dense_mat[pos2], TRMF_res2[pos2])
    TRMF_res2_rmse2 = rmse(dense_mat[pos2], TRMF_res2[pos2])

    print("TRMF_res2_mape2", TRMF_res2_mape2)
    print("TRMF_res2_rmse2", TRMF_res2_rmse2)


if __name__ == "__main__":
    test_TRMF()

    # TRMF_res2_mape2
    # 5.957667699573556
    # TRMF_res2_rmse2
    # 3.7104503211466895
