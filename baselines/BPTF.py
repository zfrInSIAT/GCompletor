import numpy as np
import pandas as pd
import tqdm
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.random import multivariate_normal as mvnrnd
from numpy.random import normal as normrnd
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import khatri_rao as kr_prod
from scipy.linalg import solve_triangular as solve_ut
from scipy.stats import wishart

from .metrics import rmse, mape


def cp_combine(U, V, X):
    return np.einsum('is, js, ts -> ijt', U, V, X)


def mvnrnd_pre(mu, Lambda):
    src = normrnd(size=(mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
                    src, lower=False, check_finite=False, overwrite_b=True) + mu


def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def sample_factor_u(tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
    """Sampling M-by-R factor matrix U and its hyperparameters (mu_u, Lambda_u)."""

    dim1, rank = U.shape
    U_bar = np.mean(U, axis=0)
    temp = dim1 / (dim1 + beta0)
    var_mu_hyper = temp * U_bar
    var_U_hyper = inv(np.eye(rank) + cov_mat(U, U_bar) + temp * beta0 * np.outer(U_bar, U_bar))
    var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_U_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim1 + beta0) * var_Lambda_hyper)

    var1 = kr_prod(X, V).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 0).T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
    var4 = var1 @ ten2mat(tau_sparse_tensor, 0).T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
    for i in range(dim1):
        U[i, :] = (solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

    return U


def sample_factor_v(tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
    """Sampling N-by-R factor matrix V and its hyperparameters (mu_v, Lambda_v)."""

    dim2, rank = V.shape
    V_bar = np.mean(V, axis=0)
    temp = dim2 / (dim2 + beta0)
    var_mu_hyper = temp * V_bar
    var_V_hyper = inv(np.eye(rank) + cov_mat(V, V_bar) + temp * beta0 * np.outer(V_bar, V_bar))
    var_Lambda_hyper = wishart.rvs(df=dim2 + rank, scale=var_V_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim2 + beta0) * var_Lambda_hyper)

    var1 = kr_prod(X, U).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 1).T).reshape([rank, rank, dim2]) + var_Lambda_hyper[:, :, None]
    var4 = var1 @ ten2mat(tau_sparse_tensor, 1).T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
    for j in range(dim2):
        V[j, :] = mvnrnd_pre(solve(var3[:, :, j], var4[:, j]), var3[:, :, j])

    return V


def sample_factor_x(tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
    """Sampling T-by-R factor matrix X and its hyperparameters."""

    dim3, rank = X.shape
    var_mu_hyper = X[0, :] / (beta0 + 1)
    dx = X[1:, :] - X[: -1, :]
    var_V_hyper = inv(np.eye(rank) + dx.T @ dx + beta0 * np.outer(X[0, :], X[0, :]) / (beta0 + 1))
    var_Lambda_hyper = wishart.rvs(df=dim3 + rank, scale=var_V_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (beta0 + 1) * var_Lambda_hyper)

    var1 = kr_prod(V, U).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 2).T).reshape([rank, rank, dim3])
    var4 = var1 @ ten2mat(tau_sparse_tensor, 2).T
    for t in range(dim3):
        if t == 0:
            X[t, :] = mvnrnd_pre((X[t + 1, :] + var_mu_hyper) / 2, var3[:, :, t] + 2 * var_Lambda_hyper)
        elif t == dim3 - 1:
            temp1 = var4[:, t] + var_Lambda_hyper @ X[t - 1, :]
            temp2 = var3[:, :, t] + var_Lambda_hyper
            X[t, :] = mvnrnd_pre(solve(temp2, temp1), temp2)
        else:
            temp1 = var4[:, t] + var_Lambda_hyper @ (X[t - 1, :] + X[t + 1, :])
            temp2 = var3[:, :, t] + 2 * var_Lambda_hyper
            X[t, :] = mvnrnd_pre(solve(temp2, temp1), temp2)

    return X


def sample_precision_tau(sparse_tensor, tensor_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
    return np.random.gamma(var_alpha, 1 / var_beta)


def BPTF(sparse_tensor_ori, rank=50, burn_iter=1000, gibbs_iter=50, period=7):
    """Bayesian Probabilistic Tensor Factorization, BPTF."""
    sparse_tensor = sparse_tensor_ori.copy()
    # dim = np.array(sparse_tensor.shape)
    dim = sparse_tensor.shape
    # 为了让算法比较好适应从用户界面接收数据，增加几行处理代码******
    # 当传入的参数为矩阵时，需要reshape
    sparse_tensor = np.reshape(sparse_tensor, [dim[0], dim[1] // period, period])

    dim1, dim2, dim3 = sparse_tensor.shape
    # 这里结束*******
    binary_tensor = np.zeros((dim1, dim2, dim3))
    dim = np.array([dim1, dim2, dim3])
    U = 0.1 * np.random.rand(dim1, rank)
    V = 0.1 * np.random.rand(dim2, rank)
    X = 0.1 * np.random.rand(dim3, rank)
    ind = None
    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0

    elif np.isnan(sparse_tensor).any() == True:
        ind = ~np.isnan(sparse_tensor)
        sparse_tensor[np.isnan(sparse_tensor)] = 0

    tau = 1

    tensor_hat_plus = np.zeros(dim)
    for it in tqdm.trange(burn_iter + gibbs_iter):
        tau_ind = tau * ind
        tau_sparse_tensor = tau * sparse_tensor
        U = sample_factor_u(tau_sparse_tensor, tau_ind, U, V, X)
        V = sample_factor_v(tau_sparse_tensor, tau_ind, U, V, X)
        X = sample_factor_x(tau_sparse_tensor, tau_ind, U, V, X)
        tensor_hat = np.einsum('is, js, ts -> ijt', U, V, X)

        tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)
        if it + 1 > burn_iter:
            tensor_hat_plus += tensor_hat

    tensor_hat = tensor_hat_plus / gibbs_iter

    return tensor_hat.reshape([dim1,-1])


# def BPTF(sparse_tensor_ori,  rank=50, burn_iter=1000, gibbs_iter=100):
# 
#     """Bayesian probabilistic tensor factorization."""
#     """Bayesian Probabilistic Tensor Factorization, BPTF."""
# 
# 
#     sparse_tensor=sparse_tensor_ori.copy()
#     dim1, dim2, dim3 = sparse_tensor.shape
#     binary_tensor = np.zeros((dim1, dim2, dim3))
#     dim = np.array([dim1, dim2, dim3])
#     # pos = np.where((dense_tensor != 0) & (sparse_tensor == 0))
#     position = np.where(sparse_tensor != 0)
#     binary_tensor[position] = 1
# 
#     # U = init["U"]
#     # V = init["V"]
#     # X = init["X"]
#     # init = {"U": 0.1 * np.random.rand(dim1, rank),
#     #         "V": 0.1 * np.random.rand(dim2, rank),
#     #         "X": 0.1 * np.random.rand(dim3, rank)}
#     U=0.1 * np.random.rand(dim1, rank)
#     V= 0.1 * np.random.rand(dim2, rank)
#     X=0.1 * np.random.rand(dim3, rank)
# 
#     beta0 = 1
#     nu0 = rank
#     mu0 = np.zeros((rank))
#     W0 = np.eye(rank)
#     tau = 1
#     alpha = 1e-6
#     beta = 1e-6
#     rho = 0.1 * np.zeros((rank))
# 
#     U_plus = np.zeros((dim1, rank))
#     V_plus = np.zeros((dim2, rank))
#     X_plus = np.zeros((dim3, rank))
#     tensor_hat_plus = np.zeros((dim1, dim2, dim3))
#     for iters in tqdm.trange(burn_iter):
#         for order in range(2):
#             if order == 0:
#                 mat = U.copy()
#             elif order == 1:
#                 mat = V.copy()
#             mat_bar = np.mean(mat, axis=0)
#             var_mu_hyper = (dim[order] * mat_bar + beta0 * mu0) / (dim[order] + beta0)
#             var_W_hyper = inv(inv(W0) + cov_mat(mat) + dim[order] * beta0 / (dim[order] + beta0)
#                               * np.outer(mat_bar - mu0, mat_bar - mu0))
#             var_Lambda_hyper = wishart(df=dim[order] + nu0, scale=var_W_hyper, seed=None).rvs()
#             var_mu_hyper = mvnrnd(var_mu_hyper, inv((dim[order] + beta0) * var_Lambda_hyper))
# 
#             if order == 0:
#                 var1 = kr_prod(X, V).T
#             elif order == 1:
#                 var1 = kr_prod(X, U).T
#             var2 = kr_prod(var1, var1)
#             var3 = (tau * np.matmul(var2, ten2mat(binary_tensor, order).T).reshape([rank, rank, dim[order]])
#                     + np.dstack([var_Lambda_hyper] * dim[order]))
#             var4 = (tau * np.matmul(var1, ten2mat(sparse_tensor, order).T)
#                     + np.dstack([np.matmul(var_Lambda_hyper, var_mu_hyper)] * dim[order])[0, :, :])
#             for i in range(dim[order]):
#                 var_Lambda = var3[:, :, i]
#                 inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2)
#                 vec = mvnrnd(np.matmul(inv_var_Lambda, var4[:, i]), inv_var_Lambda)
#                 if order == 0:
#                     U[i, :] = vec.copy()
#                 elif order == 1:
#                     V[i, :] = vec.copy()
# 
#         var_mu_hyper = (beta0 * rho + X[0, :]) / (beta0 + 1)
#         var_W_hyper = inv(inv(W0) + np.matmul((X[1: dim3, :] - X[0: dim3 - 1, :]).T,
#                                               X[1: dim3, :] - X[0: dim3 - 1, :])
#                           + (beta0 * np.outer(X[0, :] - rho, X[0, :] - rho)) / (1 + beta0))
#         var_Lambda_hyper = wishart(df=dim3 + nu0, scale=var_W_hyper, seed=None).rvs()
#         var_mu_hyper = mvnrnd(var_mu_hyper, inv((1 + beta0) * var_Lambda_hyper))
# 
#         var1 = kr_prod(V, U).T
#         var2 = kr_prod(var1, var1)
#         var3 = (tau * np.matmul(var2, ten2mat(binary_tensor, 2).T).reshape([rank, rank, dim3])
#                 + np.dstack([var_Lambda_hyper] * dim3))
#         var4 = tau * np.matmul(var1, ten2mat(sparse_tensor, 2).T)
#         for t in range(dim3):
#             if t == 0:
#                 var_mu = (X[t + 1, :] + var_mu_hyper) / 2
#                 var_Lambda = var_Lambda_hyper + var3[:, :, t]
#                 inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2)
#             elif t == dim3 - 1:
#                 inv_var_Lambda = inv((var3[:, :, t] + var3[:, :, t].T) / 2)
#                 var_mu = np.matmul(inv_var_Lambda, var4[:, t] + np.matmul(var_Lambda_hyper, X[t - 1, :]))
#             else:
#                 var_Lambda = var_Lambda_hyper + var3[:, :, t]
#                 inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2)
#                 var_mu = np.matmul(inv_var_Lambda, var4[:, t]
#                                    + np.matmul(var_Lambda_hyper, X[t + 1, :] + X[t - 1, :]))
#             X[t, :] = mvnrnd(var_mu, inv_var_Lambda)
# 
#         if iters + 1 > burn_iter - gibbs_iter:
#             U_plus += U
#             V_plus += V
#             X_plus += X
# 
#         tensor_hat = cp_combine(U, V, X)
#         if iters + 1 > burn_iter - gibbs_iter:
#             tensor_hat_plus += tensor_hat
#         # rmse = np.sqrt(np.sum((dense_tensor[pos] - tensor_hat[pos]) ** 2) / dense_tensor[pos].shape[0])
# 
#         var_alpha = alpha + 0.5 * sparse_tensor[position].shape[0]
#         error = sparse_tensor - tensor_hat
#         var_beta = beta + 0.5 * np.sum(error[position] ** 2)
#         tau = np.random.gamma(var_alpha, 1 / var_beta)
# 
#     tensor_hat = tensor_hat_plus / gibbs_iter
# 
# 
#     return tensor_hat

def test_BPTF():
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

    BPTF_res2 = BPTF(sparse_tensor2, 100, 20).reshape(dense_mat.shape)

    BPTF_res2_mape2 = mape(dense_mat[pos2], BPTF_res2[pos2])
    BPTF_res2_rmse2 = rmse(dense_mat[pos2], BPTF_res2[pos2])

    print("BPTF_res2_mape2", BPTF_res2_mape2)
    print("BPTF_res2_rmse2", BPTF_res2_rmse2)

if __name__=="__main__":
    test_BPTF()

    # BPTF_res2_mape2
    # 6.946183875673493
    # BPTF_res2_rmse2
    # 4.3508063966209685