import numpy as np
import tqdm
import pandas as pd
from .metrics import mape,rmse
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from numpy.linalg import solve as solve
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
def mvnrnd_pre(mu, Lambda):
    src = normrnd(size=(mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
                    src, lower=False, check_finite=False, overwrite_b=True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def sample_precision_tau(sparse_mat, mat_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind)
    return np.random.gamma(var_alpha, 1 / var_beta)

# def sample_factor_w(tau_sparse_mat,sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0):
def sample_factor_w(tau_sparse_mat,  tau_ind, W, X, tau, beta0=1, vargin=0):

    """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""
    dim1, rank = W.shape
    W_bar = np.mean(W, axis=0)
    temp = dim1 / (dim1 + beta0)
    var_mu_hyper = temp * W_bar
    var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
    var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_W_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim1 + beta0) * var_Lambda_hyper)
    if dim1 * rank ** 2 > 1e+8:
        vargin = 1
    if vargin == 0:
        var1 = X.T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, np.newaxis]
        var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
        for i in range(dim1):
            W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    elif vargin == 1:
        for i in range(dim1):
            pos0 = np.where(tau_sparse_mat[i, :] != 0)
            Xt = X[pos0[0], :]
            var_mu = tau * Xt.T @ tau_sparse_mat[i, pos0[0]] + var_Lambda_hyper @ var_mu_hyper
            var_Lambda = tau * Xt.T @ Xt + var_Lambda_hyper
            W[i, :] = mvnrnd_pre(solve(var_Lambda, var_mu), var_Lambda)
    return W

def sample_factor_x(tau_sparse_mat, tau_ind, W, X, beta0=1):
    """Sampling T-by-R factor matrix X and its hyperparameters (mu_x, Lambda_x)."""
    dim2, rank = X.shape
    X_bar = np.mean(X, axis=0)
    temp = dim2 / (dim2 + beta0)
    var_mu_hyper = temp * X_bar
    var_X_hyper = inv(np.eye(rank) + cov_mat(X, X_bar) + temp * beta0 * np.outer(X_bar, X_bar))
    var_Lambda_hyper = wishart.rvs(df=dim2 + rank, scale=var_X_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim2 + beta0) * var_Lambda_hyper)
    var1 = W.T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + var_Lambda_hyper[:, :, np.newaxis]
    var4 = var1 @ tau_sparse_mat + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    for t in range(dim2):
        X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t]), var3[:, :, t])
    return X

def BPMF( sparse_mat_ori, rank=50, burn_iter=1000, gibbs_iter=200):
    """Bayesian Probabilistic Matrix Factorization, BPMF."""

    sparse_mat = sparse_mat_ori.copy()
    dim1, dim2 = sparse_mat.shape
    W = 0.1 * np.random.randn(dim1, rank)
    X = 0.1 * np.random.randn(dim2, rank)
    ind=None
    if np.isnan(sparse_mat).any() == False:
        ind = sparse_mat != 0
    elif np.isnan(sparse_mat).any() == True:
        ind = ~np.isnan(sparse_mat)
        sparse_mat[np.isnan(sparse_mat)] = 0
    tau = 1
    W_plus = np.zeros((dim1, rank))
    X_plus = np.zeros((dim2, rank))
    temp_hat = np.zeros(sparse_mat.shape)
    mat_hat_plus = np.zeros(sparse_mat.shape)
    for it in tqdm.trange(burn_iter + gibbs_iter):
        tau_ind = tau * ind
        tau_sparse_mat = tau * sparse_mat
        W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
        X = sample_factor_x(tau_sparse_mat, tau_ind, W, X)
        mat_hat = W @ X.T
        tau = sample_precision_tau(sparse_mat, mat_hat, ind)
        temp_hat += mat_hat
        if it + 1 > burn_iter:
            W_plus += W
            X_plus += X
            mat_hat_plus += mat_hat
    mat_hat = mat_hat_plus / gibbs_iter
    return mat_hat

def test_BPMF():
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

    BPMF_res2 = BPMF(sparse_mat2, 100, 20)

    BPMF_res2_mape2 = mape(dense_mat[pos2], BPMF_res2[pos2])
    BPMF_res2_rmse2 = rmse(dense_mat[pos2], BPMF_res2[pos2])

    print("BPMF_res2_mape2", BPMF_res2_mape2)
    print("BPMF_res2_rmse2", BPMF_res2_rmse2)

if __name__=='__main__':
    test_BPMF()

    # BPMF_res2_mape2
    # 5.811093220252501
    # BPMF_res2_rmse2
    # 3.7058343169943675

    # 修改之后
    # BPMF_res2_mape2
    # 5.794215234510841
    # BPMF_res2_rmse2
    # 3.6956876775794303