import numpy as np
import pandas as pd
import tqdm
from .metrics import rmse,mape
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from numpy.linalg import inv as inv
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


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def cp_combine(var):
    return np.einsum('is, js, ts -> ijt', var[0], var[1], var[2])


def vec_combine(vector):
    tensor = 0
    d = len(vector)
    for i in range(d):
        ax = [len(vector[i]) if j == i else 1 for j in range(d)]
        tensor = tensor + vector[i].reshape(ax, order='F')
    return tensor


def sample_global_mu(mu_sparse, pos_obs, tau_eps, tau0=1):
    tau_tilde = 1 / (tau_eps * len(pos_obs[0]) + tau0)
    mu_tilde = tau_eps * np.sum(mu_sparse) * tau_tilde
    return np.random.normal(mu_tilde, np.sqrt(tau_tilde))


def sample_bias_vector(bias_sparse, factor, bias, ind, dim, k, tau_eps, tau0=1):
    for k in range(len(dim)):
        idx = tuple(filter(lambda x: x != k, range(len(dim))))
        temp = factor.copy()
        temp[k] = np.zeros((dim[k]))
        tau_tilde = 1 / (tau_eps * bias[k] + tau0)
        mu_tilde = tau_eps * np.sum(ind * (bias_sparse - vec_combine(temp)), axis=idx) * tau_tilde
        factor[k] = np.random.normal(mu_tilde, np.sqrt(tau_tilde))
    return factor


def sample_factor(tau_sparse, factor, ind, dim, k, tau_eps, beta0=1):
    dim, rank = factor[k].shape
    dim = factor[k].shape[0]
    factor_bar = np.mean(factor[k], axis=0)
    temp = dim / (dim + beta0)
    var_mu_hyper = temp * factor_bar
    var_W_hyper = inv(
        np.eye(rank) + cov_mat(factor[k], factor_bar) + temp * beta0 * np.outer(factor_bar, factor_bar))
    var_Lambda_hyper = wishart.rvs(df=dim + rank, scale=var_W_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_Lambda_hyper)

    idx = list(filter(lambda x: x != k, range(len(factor))))
    var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_eps * ind, k).T).reshape([rank, rank, dim]) + var_Lambda_hyper[:, :, np.newaxis]
    var4 = var1 @ ten2mat(tau_sparse, k).T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    for i in range(dim):
        factor[k][i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return factor[k]


def sample_precision_tau(error_tensor, pos_obs):
    var_alpha = 1e-6 + 0.5 * len(pos_obs[0])
    var_beta = 1e-6 + 0.5 * np.linalg.norm(error_tensor, 2) ** 2
    return np.random.gamma(var_alpha, 1 / var_beta)

def BATF_Gibbs(sparse_tensor_ori, rank = 50,burn_iter=1000, gibbs_iter=200):
    """Bayesian Augmented Tensor Factorization (BATF) with Gibbs sampling.
    sparse_tensor_ori:3d tensor
    """


    # if copy:
    #     sparse_tensor = sparse_tensor_ori.copy()
    # else:
    #     sparse_tensor = sparse_tensor_ori
    sparse_tensor = sparse_tensor_ori.copy()
    dim = np.array(sparse_tensor.shape)

    vector = []
    factor = []
    for k in range(len(dim)):
        vector.append(0.1 * np.random.randn(dim[k], ))
        factor.append(0.1 * np.random.randn(dim[k], rank))
    dim = np.array(sparse_tensor.shape)
    ind=None
    pos_obs=None
    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0
        pos_obs = np.where(ind)
    elif np.isnan(sparse_tensor).any() == True:
        ind = ~np.isnan(sparse_tensor)
        pos_obs = np.where(ind)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    tau_eps = 1
    bias = []
    for k in range(len(dim)):
        idx = tuple(filter(lambda x: x != k, range(len(dim))))
        bias.append(np.sum(ind, axis=idx))
    temp = cp_combine(factor)
    tensor_hat_plus = np.zeros(dim)
    for it in tqdm.trange(burn_iter + gibbs_iter):
        temp = sparse_tensor - temp
        mu_glb = sample_global_mu(temp[pos_obs] - vec_combine(vector)[pos_obs], pos_obs, tau_eps)
        vector = sample_bias_vector(temp - mu_glb, vector, bias, ind, dim, k, tau_eps)
        del temp
        tau_sparse = tau_eps * ind * (sparse_tensor - mu_glb - vec_combine(vector))
        for k in range(len(dim)):
            factor[k] = sample_factor(tau_sparse, factor, ind, dim, k, tau_eps)
        temp = cp_combine(factor)
        tensor_hat = mu_glb + vec_combine(vector) + temp
        tau_eps = sample_precision_tau(sparse_tensor[pos_obs] - tensor_hat[pos_obs], pos_obs)
        if it + 1 > burn_iter:
            tensor_hat_plus += tensor_hat
    tensor_hat = tensor_hat_plus / gibbs_iter
    return tensor_hat

def test_BATF():
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

    BATF_Gibbs_res2 = BATF_Gibbs(sparse_tensor2,burn_iter=100, gibbs_iter=20).reshape(dense_mat.shape)

    BATF_Gibbs_mape2 = mape(dense_mat[pos2], BATF_Gibbs_res2[pos2])
    BATF_Gibbs_rmse2 = rmse(dense_mat[pos2], BATF_Gibbs_res2[pos2])

    print("BATF_Gibbs_mape2", BATF_Gibbs_mape2)
    print("BATF_Gibbs_rmse2", BATF_Gibbs_rmse2)

if __name__=='__main__':
    test_BATF()
