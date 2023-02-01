
import tqdm
import pandas as pd
from .metrics import rmse,mape
import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import matplotlib.pyplot as plt

def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False),
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')


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
        U[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

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


def sample_theta(X, theta, Lambda_x, time_lags, beta0=1):
    dim, rank = X.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)
    theta_bar = np.mean(theta, axis=0)
    temp = d / (d + beta0)
    var_theta_hyper = inv(np.eye(rank) + cov_mat(theta, theta_bar)
                          + temp * beta0 * np.outer(theta_bar, theta_bar))
    var_Lambda_hyper = wishart.rvs(df=d + rank, scale=var_theta_hyper)
    var_mu_hyper = mvnrnd_pre(temp * theta_bar, (d + beta0) * var_Lambda_hyper)

    for k in range(d):
        theta0 = theta.copy()
        theta0[k, :] = 0
        mat0 = np.zeros((dim - tmax, rank))
        for L in range(d):
            mat0 += X[tmax - time_lags[L]: dim - time_lags[L], :] @ np.diag(theta0[L, :])
        varPi = X[tmax: dim, :] - mat0
        var0 = X[tmax - time_lags[k]: dim - time_lags[k], :]
        var = np.einsum('ij, jk, ik -> j', var0, Lambda_x, varPi)
        var_Lambda = np.einsum('ti, tj, ij -> ij', var0, var0, Lambda_x) + var_Lambda_hyper
        theta[k, :] = mvnrnd_pre(solve(var_Lambda, var + var_Lambda_hyper @ var_mu_hyper), var_Lambda)

    return theta


def sample_Lambda_x(X, theta, time_lags):
    dim, rank = X.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)
    mat = X[: tmax, :].T @ X[: tmax, :]
    temp = np.zeros((dim - tmax, rank, d))
    for k in range(d):
        temp[:, :, k] = X[tmax - time_lags[k]: dim - time_lags[k], :]
    new_mat = X[tmax: dim, :] - np.einsum('kr, irk -> ir', theta, temp)
    Lambda_x = wishart.rvs(df=dim + rank, scale=inv(np.eye(rank) + mat + new_mat.T @ new_mat))

    return Lambda_x


def sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, U, V, X, theta, Lambda_x):
    """Sampling T-by-R factor matrix X."""

    dim3, rank = X.shape
    tmax = np.max(time_lags)
    tmin = np.min(time_lags)
    d = time_lags.shape[0]
    A = np.zeros((d * rank, rank))
    for k in range(d):
        A[k * rank: (k + 1) * rank, :] = np.diag(theta[k, :])
    A0 = np.dstack([A] * d)
    for k in range(d):
        A0[k * rank: (k + 1) * rank, :, k] = 0
    mat0 = Lambda_x @ A.T
    mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
    mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))

    var1 = kr_prod(V, U).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 2).T).reshape([rank, rank, dim3]) + Lambda_x[:, :, None]
    var4 = var1 @ ten2mat(tau_sparse_tensor, 2).T
    for t in range(dim3):
        Mt = np.zeros((rank, rank))
        Nt = np.zeros(rank)
        Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
        index = list(range(0, d))
        if t >= dim3 - tmax and t < dim3 - tmin:
            index = list(np.where(t + time_lags < dim3))[0]
        elif t < tmax:
            Qt = np.zeros(rank)
            index = list(np.where(t + time_lags >= tmax))[0]
        if t < dim3 - tmin:
            Mt = mat2.copy()
            temp = np.zeros((rank * d, len(index)))
            n = 0
            for k in index:
                temp[:, n] = X[t + time_lags[k] - time_lags, :].reshape(rank * d)
                n += 1
            temp0 = X[t + time_lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
            Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)

        var3[:, :, t] = var3[:, :, t] + Mt
        if t < tmax:
            var3[:, :, t] = var3[:, :, t] - Lambda_x + np.eye(rank)
        X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])

    return X


def BTRTF( sparse_tensor_ori,  rank=30, time_lags=(1,2,24),
           burn_iter=1100, gibbs_iter=100, period=7):
    """Bayesian Temporal Regularized Tensor Factorization, BTRTF."""

    sparse_tensor = sparse_tensor_ori.copy()
    # ****
    dim = sparse_tensor.shape
    sparse_tensor = np.reshape(sparse_tensor,[dim[0], dim[1]//period, period])
    dim1, dim2, dim3 = sparse_tensor.shape
    time_lags = np.array(time_lags)
    d = time_lags.shape[0]

    U = 0.1 * np.random.rand(dim1, rank)
    V = 0.1 * np.random.rand(dim2, rank)
    X = 0.1 * np.random.rand(dim3, rank)
    theta = 0.1 * np.random.rand(d, rank)

    ind=None
    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0
        
    elif np.isnan(sparse_tensor).any() == True:
        
        ind = ~np.isnan(sparse_tensor)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    U_plus = np.zeros((dim1, rank))
    V_plus = np.zeros((dim2, rank))
    X_plus = np.zeros((dim3, rank))
    theta_plus = np.zeros((d, rank))
    
    tau = 1
    tensor_hat_plus = np.zeros(sparse_tensor.shape)
    for it in tqdm.trange(burn_iter + gibbs_iter):
        tau_ind = tau * ind
        tau_sparse_tensor = tau * sparse_tensor
        U = sample_factor_u(tau_sparse_tensor, tau_ind, U, V, X)
        V = sample_factor_v(tau_sparse_tensor, tau_ind, U, V, X)
        Lambda_x = sample_Lambda_x(X, theta, time_lags)
        theta = sample_theta(X, theta, Lambda_x, time_lags)
        X = sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, U, V, X, theta, Lambda_x)
        tensor_hat = np.einsum('is, js, ts -> ijt', U, V, X)
        tau = np.random.gamma(1e-6 + 0.5 * np.sum(ind),
                              1 / (1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)))
        
        if it + 1 > burn_iter:
            # U_plus += U
            # V_plus += V
            # X_plus += X
            # theta_plus += theta
            tensor_hat_plus += tensor_hat
    tensor_hat = tensor_hat_plus / gibbs_iter

    return tensor_hat.reshape([dim1, dim2*dim3])

def test_BTRTF():
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
    BTRTF_res2 = BTRTF(sparse_tensor2, rank=50,time_lags=(1,2,288),burn_iter=100, gibbs_iter=20).reshape(dense_mat.shape)

    BTRTF_res2_mape2 = mape(dense_mat[pos2], BTRTF_res2[pos2])
    BTRTF_res2_rmse2 = rmse(dense_mat[pos2], BTRTF_res2[pos2])

    print("BTRTF_res2_mape2", BTRTF_res2_mape2)
    print("BTRTF_res2_rmse2", BTRTF_res2_rmse2)


if __name__ == "__main__":
    test_BTRTF()

    # BTRTF_res2_mape2
    # 7.75779098440859
    # BTRTF_res2_rmse2
    # 4.645867823470987