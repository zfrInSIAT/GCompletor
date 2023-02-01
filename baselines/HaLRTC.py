import numpy as np
from numpy.linalg import inv as inv
import pandas as pd
from .metrics import rmse,mape
# import tqdm
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order='F'), 0, mode)


def svt(mat, tau):
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=False)
        s = np.sqrt(s)
        tol = n * np.finfo(float).eps * np.max(s)
        idx = np.sum(s > max(tau, tol))
        mid = (s[:idx] - tau) / s[:idx]
        return u[:, :idx] @ np.diag(mid) @ u[:, :idx].T @ mat
    elif m > 2 * n:
        return svt(mat.T, tau).T
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    idx = np.sum(s > tau)
    return u[:, :idx] @ np.diag(s[:idx] - tau) @ v[:idx, :]


def HaLRTC(sparse_tensor_ori, rho=1e-5, epsilon=1e-4, maxiter=200):
    sparse_tensor = sparse_tensor_ori
    dim = np.array(sparse_tensor.shape)
    alpha = np.ones(3) / 3

    pos_miss=None
    if np.isnan(sparse_tensor).any() == False:
        pos_miss = np.where(sparse_tensor == 0)
    elif np.isnan(sparse_tensor).any() == True:
        sparse_tensor[np.isnan(sparse_tensor)] = 0
        pos_miss = np.where(sparse_tensor == 0)

    tensor_hat = sparse_tensor.copy()
    B = [np.zeros(sparse_tensor.shape) for _ in range(len(dim))]
    Y = [np.zeros(sparse_tensor.shape) for _ in range(len(dim))]
    last_ten = sparse_tensor.copy()
    snorm = np.linalg.norm(sparse_tensor)

    it = 0
    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            B[k] = mat2ten(svt(ten2mat(tensor_hat + Y[k] / rho, k), alpha[k] / rho), dim, k)
        tensor_hat[pos_miss] = ((sum(B) - sum(Y) / rho) / 3)[pos_miss]
        for k in range(len(dim)):
            Y[k] = Y[k] - rho * (B[k] - tensor_hat)
        tol = np.linalg.norm((tensor_hat - last_ten)) / snorm
        last_ten = tensor_hat.copy()
        it += 1
        if (tol < epsilon) or (it >= maxiter):
            break
    return tensor_hat

def test_HaLRTC():
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
    HaLRTC_res2 = HaLRTC(sparse_tensor2, rho=1e-5, epsilon=1e-4, maxiter=200).reshape(
        dense_mat.shape)

    HaLRTC_res2_mape2 = mape(dense_mat[pos2], HaLRTC_res2[pos2])
    HaLRTC_res2_rmse2 = rmse(dense_mat[pos2], HaLRTC_res2[pos2])

    print("HaLRTC_res2_mape2", HaLRTC_res2_mape2)
    print("HaLRTC_res2_rmse2", HaLRTC_res2_rmse2)


if __name__ == "__main__":
    test_HaLRTC()

    # HaLRTC_res2_mape2
    # 5.9261458014782535
    # HaLRTC_res2_rmse2
    # 3.468561835655723
