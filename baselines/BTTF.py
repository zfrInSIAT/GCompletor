import numpy as np
import pandas as pd
import tqdm
from numpy.linalg import cholesky as cholesky_lower
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.random import normal as normrnd
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import khatri_rao as kr_prod
from scipy.linalg import solve_triangular as solve_ut
from scipy.stats import invwishart
from scipy.stats import wishart

from .metrics import rmse, mape


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


def mnrnd(M, U, V):
    """
    Generate matrix normal distributed random matrix.
    M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
    """
    dim1, dim2 = M.shape
    X0 = np.random.randn(dim1, dim2)
    P = cholesky_lower(U)
    Q = cholesky_lower(V)

    return M + P @ X0 @ Q.T


def sample_var_coefficient(X, time_lags):
    dim, rank = X.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)

    Z_mat = X[tmax: dim, :]
    Q_mat = np.zeros((dim - tmax, rank * d))
    for k in range(d):
        Q_mat[:, k * rank: (k + 1) * rank] = X[tmax - time_lags[k]: dim - time_lags[k], :]
    var_Psi0 = np.eye(rank * d) + Q_mat.T @ Q_mat
    var_Psi = inv(var_Psi0)
    var_M = var_Psi @ Q_mat.T @ Z_mat
    var_S = np.eye(rank) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
    Sigma = invwishart.rvs(df=rank + dim - tmax, scale=var_S)

    return mnrnd(var_M, var_Psi, Sigma), Sigma


def sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, Lambda_x):
    """Sampling T-by-R factor matrix X."""

    dim3, rank = X.shape
    tmax = np.max(time_lags)
    tmin = np.min(time_lags)
    d = time_lags.shape[0]
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


def sample_precision_tau(sparse_tensor, tensor_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind, axis=2)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind, axis=2)
    return np.random.gamma(var_alpha, 1 / var_beta)


def BTTF(sparse_tensor_ori, rank=30, time_lags=(1, 2, 24), burn_iter=1000, gibbs_iter=200, multi_steps=1, vargin=0):
    """Bayesian Temporal Tensor Factorization, BTTF."""
    sparse_tensor = sparse_tensor_ori.copy()
    time_lags = np.array(time_lags)
    dim1, dim2, dim3 = sparse_tensor.shape
    d = time_lags.shape[0]
    U = 0.1 * np.random.randn(dim1, rank)
    V = 0.1 * np.random.randn(dim2, rank)
    X = 0.1 * np.random.randn(dim3, rank)

    # init = {"U": 0.1 * np.random.randn(dim1, rank), "V": 0.1 * np.random.randn(dim2, rank),
    #         "X": 0.1 * np.random.randn(dim3, rank)}

    ind = None
    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0

    elif np.isnan(sparse_tensor).any() == True:
        ind = ~np.isnan(sparse_tensor)
        sparse_tensor[np.isnan(sparse_tensor)] = 0

    U_plus = np.zeros((dim1, rank))
    V_plus = np.zeros((dim2, rank))
    X_new_plus = np.zeros((dim3 + multi_steps, rank))
    A_plus = np.zeros((rank * d, rank))
    tau = None
    if vargin == 0:  # scalar tau
        tau = 1
    elif vargin == 1:  # matrix tau
        tau = np.ones((dim1, dim2))
    tensor_hat_plus = np.zeros(sparse_tensor.shape)
    for it in tqdm.trange(burn_iter + gibbs_iter):
        if vargin == 0:  # scalar tau
            tau_ind = tau * ind
            tau_sparse_tensor = tau * sparse_tensor
            U = sample_factor_u(tau_sparse_tensor, tau_ind, U, V, X)
            V = sample_factor_v(tau_sparse_tensor, tau_ind, U, V, X)
            A, Sigma = sample_var_coefficient(X, time_lags)
            X = sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, inv(Sigma))
            tensor_hat = np.einsum('is, js, ts -> ijt', U, V, X)
            tau = np.random.gamma(1e-6 + 0.5 * np.sum(ind),
                                  1 / (1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)))
        elif vargin == 1:  # matrix tau
            tau_ind = tau[:, :, None] * ind
            tau_sparse_tensor = tau[:, :, None] * sparse_tensor
            U = sample_factor_u(tau_sparse_tensor, tau_ind, U, V, X)
            V = sample_factor_v(tau_sparse_tensor, tau_ind, U, V, X)
            A, Sigma = sample_var_coefficient(X, time_lags)
            X = sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, U, V, X, A, inv(Sigma))
            tensor_hat = np.einsum('is, js, ts -> ijt', U, V, X)
            tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)
        X_new = np.zeros((dim3 + multi_steps, rank))
        if it + 1 > burn_iter:
            U_plus += U
            V_plus += V
            A_plus += A
            X_new[: dim3, :] = X.copy()
            if multi_steps == 1:
                X_new[dim3, :] = A.T @ X_new[dim3 - time_lags, :].reshape(rank * d)
            elif multi_steps > 1:
                for t0 in range(multi_steps):
                    X_new[dim3 + t0, :] = A.T @ X_new[dim3 + t0 - time_lags, :].reshape(rank * d)
            X_new_plus += X_new
            tensor_hat_plus += tensor_hat
    tensor_hat = tensor_hat_plus / gibbs_iter

    return tensor_hat


def test_BTTF():
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
    BTTF_res2 = BTTF(sparse_tensor2, rank=50, time_lags=(1, 2, 288), burn_iter=100, gibbs_iter=20).reshape(
        dense_mat.shape)

    BTTF_res2_mape2 = mape(dense_mat[pos2], BTTF_res2[pos2])
    BTTF_res2_rmse2 = rmse(dense_mat[pos2], BTTF_res2[pos2])

    print("BTTF_res2_mape2", BTTF_res2_mape2)
    print("BTTF_res2_rmse2", BTTF_res2_rmse2)


if __name__ == "__main__":
    test_BTTF()

# BTTF_res2_mape2 7.564754829178373
# BTTF_res2_rmse2 4.547940181828565