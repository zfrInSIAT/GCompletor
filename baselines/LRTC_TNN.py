import numpy as np
from .metrics import rmse,mape
import pandas as pd
# import tqdm
from numpy.linalg import inv as inv
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order='F'), 0, mode)

# def svt_tnn(mat, alpha, rho, theta):
#     """This is a Numpy dependent singular value thresholding (SVT) process."""
#     u, s, v = np.linalg.svd(mat, full_matrices=False)
#     vec = s.copy()
#     vec[theta:] = s[theta:] - alpha / rho
#     vec[vec < 0] = 0
#     return np.matmul(np.matmul(u, np.diag(vec)), v)

def svt_tnn(mat, alpha, rho, theta):
    """This is a Numpy dependent singular value thresholding (SVT) process."""
    u, s, v = np.linalg.svd(mat, full_matrices = False)
    vec = s.copy()
    vec[theta :] = s[theta :] - alpha / rho
    vec[vec < 0] = 0
    return np.matmul(np.matmul(u, np.diag(vec)), v)

def LRTC_TNN(sparse_tensor_ori,
         rho=1e-5,
         theta=0.30,
         epsilon=1e-5,
         maxiter=200):
    """Low-Rank Tenor Completion with Truncated Nuclear Norm, LRTC-TNN.

    """

    sparse_tensor = sparse_tensor_ori.copy()
    dim = np.array(sparse_tensor.shape)

    pos_missing = None
    if np.isnan(sparse_tensor).any() == False:
        pos_missing = sparse_tensor == 0

    elif np.isnan(sparse_tensor).any() == True:

        pos_missing = np.isnan(sparse_tensor)
        sparse_tensor[np.isnan(sparse_tensor)] = 0

    alpha = np.ones(3) / 3

    X = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{X}}
    T = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{T}}
    Z = sparse_tensor.copy()
    last_tensor = sparse_tensor.copy()
    snorm = np.sqrt(np.sum(sparse_tensor ** 2))
    it = 0
    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            X[k] = mat2ten(svt_tnn(ten2mat(Z - T[k] / rho, k), alpha[k], rho, np.int(np.ceil(theta * dim[k]))), dim, k)
        Z[pos_missing] = np.mean(X + T / rho, axis=0)[pos_missing]
        T = T + rho * (X - np.broadcast_to(Z, np.insert(dim, 0, len(dim))))
        tensor_hat = np.einsum('k, kmnt -> mnt', alpha, X)
        tol = np.sqrt(np.sum((tensor_hat - last_tensor) ** 2)) / snorm
        last_tensor = tensor_hat.copy()
        it += 1
        if (tol < epsilon) or (it >= maxiter):
            break
    return tensor_hat

def test_LRTC_TNN():
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
    LRTC_TNN_res2 = LRTC_TNN(sparse_tensor2, maxiter=500).reshape(
        dense_mat.shape)

    LRTC_TNN_res2_mape2 = mape(dense_mat[pos2], LRTC_TNN_res2[pos2])
    LRTC_TNN_res2_rmse2 = rmse(dense_mat[pos2], LRTC_TNN_res2[pos2])

    print("LRTC_TNN_res2_mape2", LRTC_TNN_res2_mape2)
    print("LRTC_TNN_res2_rmse2", LRTC_TNN_res2_rmse2)


if __name__ == "__main__":
    test_LRTC_TNN()

    # LRTC_TNN_res2_mape2
    # 4.654310845754401
    # LRTC_TNN_res2_rmse2
    # 3.0619884282331524