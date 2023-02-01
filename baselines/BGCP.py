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


def mvnrnd_pre(mu, Lambda):
    # normal()函数生成一个size大小的np.array,采用正态分布随机采样,默认均值mu=0,标准差sigma=1.0
    src = normrnd(size=(mu.shape[0],))
    # solve_triangular() Solve the equation a x = b for x, assuming a is a triangular matrix.
    # lower = False,使用上三角,check_finite=False,不做无穷大检查,能提升运行速度,overwrite_b=True,允许对b参数进行重写.

    # Compute the Cholesky decomposition of a matrix.
    # Returns the Cholesky decomposition, A=LL* or A=U*U  of a Hermitian positive-definite matrix A.
    return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
                    src, lower=False, check_finite=False, overwrite_b=True) + mu


def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat


def cp_combine(var):
    return np.einsum('is, js, ts -> ijt', var[0], var[1], var[2])


def sample_precision_tau(sparse_tensor, tensor_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
    return np.random.gamma(var_alpha, 1 / var_beta)


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def sample_factor(tau_sparse_tensor, tau_ind, factor, k, beta0=1):
    dim, rank = factor[k].shape
    dim = factor[k].shape[0]
    factor_bar = np.mean(factor[k], axis=0)
    temp = dim / (dim + beta0)
    var_mu_hyper = temp * factor_bar
    # inv计算逆矩阵
    var_W_hyper = inv(
        np.eye(rank) + cov_mat(factor[k], factor_bar) + temp * beta0 * np.outer(factor_bar, factor_bar))
    # 从一个wishart分布中随机抽取样本,df为自由度，必须大于或等于尺度矩阵的维数,scale为对称正定比例矩阵的分布
    var_Lambda_hyper = wishart.rvs(df=dim + rank, scale=var_W_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_Lambda_hyper)

    idx = list(filter(lambda x: x != k, range(len(factor))))
    var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, k).T).reshape([rank, rank, dim]) + var_Lambda_hyper[:, :, np.newaxis]
    var4 = var1 @ ten2mat(tau_sparse_tensor, k).T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    for i in range(dim):
        # solve函数是已知A,B,求一个矩阵X,使得AX=B
        factor[k][i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return factor[k]


def BGCP(sparse_tensor_ori, rank=50, burn_iter=1000, gibbs_iter=200, period = 7):
    """Bayesian Gaussian CP (BGCP) decomposition.

    sparse_tensor:3d tensor

    """
    sparse_tensor = sparse_tensor_ori.copy()
    dim = np.array(sparse_tensor.shape)

    # 为了让算法比较好适应从用户界面接收数据，增加几行处理代码******
    # 当传入的参数为矩阵时，需要reshape
    sparse_tensor = np.reshape(sparse_tensor, [dim[0], dim[1]//period,period])
    dim = sparse_tensor.shape
    # 这里结束*******

    # rank是秩，也就是矩阵分解时的r

    factor = []
    # 初始化U，V，X，按顺序添加到factor中。
    for k in range(len(dim)):
        factor.append(0.1 * np.random.randn(dim[k], rank))

    # dim = np.array(sparse_tensor.shape)
    # rank = factor[0].shape[1]

    # 判断稀疏矩阵中空缺值的表示,统一用0表示空缺,ind标示稀疏矩阵中的非空缺值.
    ind = None
    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0
    elif np.isnan(sparse_tensor).any() == True:
        ind = ~np.isnan(sparse_tensor)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    tau = 1
    # 生成和factor同型的全0张量
    factor_plus = []
    for k in range(len(dim)):
        factor_plus.append(np.zeros((dim[k], rank)))
    temp_hat = np.zeros(dim)
    tensor_hat_plus = np.zeros(dim)
    for it in tqdm.trange(burn_iter + gibbs_iter):
        tau_ind = tau * ind
        tau_sparse_tensor = tau * sparse_tensor
        for k in range(len(dim)):
            factor[k] =   (tau_sparse_tensor, tau_ind, factor, k)
        tensor_hat = cp_combine(factor)
        temp_hat += tensor_hat
        tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)
        if it + 1 > burn_iter:
            factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
            tensor_hat_plus += tensor_hat
    tensor_hat = tensor_hat_plus / gibbs_iter
    return tensor_hat.reshape([dim[0],-1])


def test_BGCP():
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

    BGCP_res2 = BGCP(sparse_tensor2, 100, 20).reshape(dense_mat.shape)

    BGCP_res2_mape2 = mape(dense_mat[pos2], BGCP_res2[pos2])
    BGCP_res2_rmse2 = rmse(dense_mat[pos2], BGCP_res2[pos2])

    print("BGCP_res2_mape2", BGCP_res2_mape2)
    print("BGCP_res2_rmse2", BGCP_res2_rmse2)


if __name__ == "__main__":
    test_BGCP()

    # BGCP_res2_mape2
    # 6.73039522487567
    # BGCP_res2_rmse2
    # 4.142491871982514
