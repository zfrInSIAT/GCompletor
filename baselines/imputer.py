
import numpy as np



class Imputer:
    """Imputer class."""

    def knn(self,data_mat, n_neighbors=5, weights='uniform',
            metric='nan_euclidean', copy=True, add_indicator=False):
        """

        @param data: numpy 2d array,missing values are represented by np.nan
        @param n_neighbors: number of neighbors
        @return: numpy 2d array after imputed
        """
        #通过测试
        data = data_mat.copy()
        from sklearn.impute import KNNImputer
        imp = KNNImputer(n_neighbors=n_neighbors, weights=weights,
                         metric=metric, copy=copy, add_indicator=add_indicator)
        # imp = KNNImputer(n_neighbors=5)
        mat = imp.fit_transform(data)
        return mat

    def ppca(self,Y_mat,d=20,dia=False):

        from numpy import shape, isnan, nanmean, average, zeros, log, cov
        from numpy import matmul as mm
        from numpy.matlib import repmat
        from numpy.random import normal
        from numpy.linalg import inv, det, eig
        from numpy import identity as eye
        from numpy import trace as tr
        from scipy.linalg import orth
        """
           Implements probabilistic PCA for data with missing values,
           using a factorizing distribution over hidden states and hidden observations.
           Args:
               Y:   (N by D ) input numpy ndarray of data vectors
               d:   (  int  ) dimension of latent space
               dia: (boolean) if True: print objective each step
           Returns:
               ss: ( float ) isotropic variance outside subspace
               C:  (D by d ) C*C' + I*ss is covariance model, C has scaled principal directions as cols
               M:  (D by 1 ) data mean
               X:  (N by d ) expected states
               Ye: (N by D ) expected complete observations (differs from Y if data is missing)
               Based on MATLAB code from J.J. VerBeek, 2006. http://lear.inrialpes.fr/~verbeek
        """
        Y = Y_mat.copy()
        N, D = shape(Y)  # N observations in D dimensions (i.e. D is number of features, N is samples)
        threshold = 1E-4  # minimal relative change in objective function to continue
        hidden = isnan(Y)
        missing = hidden.sum()

        if (missing > 0):
            M = nanmean(Y, axis=0)
        else:
            M = average(Y, axis=0)

        Ye = Y - repmat(M, N, 1)

        if (missing > 0):
            Ye[hidden] = 0

        # initialize
        C = normal(loc=0.0, scale=1.0, size=(D, d))
        CtC = mm(C.T, C)
        X = mm(mm(Ye, C), inv(CtC))
        recon = mm(X, C.T)
        recon[hidden] = 0
        ss = np.sum((recon - Ye) ** 2) / (N * D - missing)

        count = 1
        old = np.inf

        # EM Iterations
        while (count):
            Sx = inv(eye(d) + CtC / ss)  # E-step, covariances
            ss_old = ss
            if (missing > 0):
                proj = mm(X, C.T)
                Ye[hidden] = proj[hidden]

            X = mm(mm(Ye, C), Sx / ss)  # E-step: expected values

            SumXtX = mm(X.T, X)  # M-step
            C = mm(mm(mm(Ye.T, X), (SumXtX + N * Sx).T), inv(mm((SumXtX + N * Sx), (SumXtX + N * Sx).T)))
            CtC = mm(C.T, C)
            ss = (np.sum((mm(X, C.T) - Ye) ** 2) + N * np.sum(CtC * Sx) + missing * ss_old) / (N * D)
            # transform Sx determinant into numpy float128 in order to deal with high dimensionality
            Sx_det = np.min(Sx).astype(np.float64) ** shape(Sx)[0] * det(Sx / np.min(Sx))
            objective = N * D + N * (D * log(ss) + tr(Sx) - log(Sx_det)) + tr(SumXtX) - missing * log(ss_old)

            rel_ch = np.abs(1 - objective / old)
            old = objective

            count = count + 1
            if (rel_ch < threshold and count > 5):
                count = 0
            if (dia == True):
                print('Objective: %.2f, Relative Change %.5f' % (objective, rel_ch))

        C = orth(C)
        covM = cov(mm(Ye, C).T)
        vals, vecs = eig(covM)
        ordr = np.argsort(vals)[::-1]
        vals = vals[ordr]
        vecs = vecs[:, ordr]

        C = mm(C, vecs)
        X = mm(Ye, C)

        # add data mean to expected complete data
        Ye = Ye + repmat(M, N, 1)

        # return C, ss, M, X, Ye
        return Ye

    def bn(self,data):
        #通过测试
        import statsmodels.imputation.bayes_mi as bm
        nan_mat = data.copy()
        model = bm.BayesGaussMI(nan_mat)
        model.update()
        return nan_mat
    def mean(self,data_mat):
        data = data_mat.copy()
        for i in range(data.shape[0]):      
            meanVal = np.mean(data[i,~np.isnan(data[i,:])])       
            data[i,np.isnan(data[i,:])] = meanVal
        return data

    def median(self,data_mat):
        data = data_mat.copy()
        for i in range(data.shape[0]):
            medianVal = np.median(data[i, ~np.isnan(data[i, :])])
            data[i, np.isnan(data[i, :])] = medianVal
        return data

    def mode(self,data_mat):
        data = data_mat.copy()
        import scipy.stats
        modeVal = scipy.stats.mode(data, axis=1)
        for i in range(data.shape[0]):
            data[i, np.isnan(data[i, :])] = modeVal[0][i]
        return data

    def BGCP(self,sparse_tensor_ori, burn_iter=1000, gibbs_iter=200):
        """Bayesian Gaussian CP (BGCP) decomposition.

        sparse_tensor:3d tensor

        """
        import numpy as np
        from numpy.linalg import inv as inv
        from scipy.stats import wishart
        from numpy.random import multivariate_normal as mvnrnd
        from numpy.random import normal as normrnd
        from scipy.linalg import cholesky as cholesky_upper
        from scipy.linalg import solve_triangular as solve_ut
        from numpy.linalg import solve as solve
        from scipy.linalg import khatri_rao as kr_prod
        def mvnrnd_pre(mu, Lambda):
            src = normrnd(size=(mu.shape[0],))
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
            var_W_hyper = inv(
                np.eye(rank) + cov_mat(factor[k], factor_bar) + temp * beta0 * np.outer(factor_bar, factor_bar))
            var_Lambda_hyper = wishart.rvs(df=dim + rank, scale=var_W_hyper)
            var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_Lambda_hyper)

            idx = list(filter(lambda x: x != k, range(len(factor))))
            var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
            var2 = kr_prod(var1, var1)
            var3 = (var2 @ ten2mat(tau_ind, k).T).reshape([rank, rank, dim]) + var_Lambda_hyper[:, :, np.newaxis]
            var4 = var1 @ ten2mat(tau_sparse_tensor, k).T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
            for i in range(dim):
                factor[k][i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
            return factor[k]

        sparse_tensor = sparse_tensor_ori.copy()
        dim = np.array(sparse_tensor.shape)
        rank = 50
        factor = []
        for k in range(len(dim)):
            factor.append(0.1 * np.random.randn(dim[k], rank))

        dim = np.array(sparse_tensor.shape)
        rank = factor[0].shape[1]
        if np.isnan(sparse_tensor).any() == False:
            ind = sparse_tensor != 0
        elif np.isnan(sparse_tensor).any() == True:
            ind = ~np.isnan(sparse_tensor)
            sparse_tensor[np.isnan(sparse_tensor)] = 0
        tau = 1
        factor_plus = []
        for k in range(len(dim)):
            factor_plus.append(np.zeros((dim[k], rank)))
        temp_hat = np.zeros(dim)
        tensor_hat_plus = np.zeros(dim)
        for it in range(burn_iter + gibbs_iter):
            tau_ind = tau * ind
            tau_sparse_tensor = tau * sparse_tensor
            for k in range(len(dim)):
                factor[k] = sample_factor(tau_sparse_tensor, tau_ind, factor, k)
            tensor_hat = cp_combine(factor)
            temp_hat += tensor_hat
            tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)
            if it + 1 > burn_iter:
                factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
                tensor_hat_plus += tensor_hat
        tensor_hat = tensor_hat_plus / gibbs_iter
        return tensor_hat

    def BATF_Gibbs(self,sparse_tensor_ori, burn_iter=1000, gibbs_iter=200):
        """Bayesian Augmented Tensor Factorization (BATF) with Gibbs sampling.
        sparse_tensor_ori:3d tensor
        """
        import numpy as np
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
                temp = vector.copy()
                temp[k] = np.zeros((dim[k]))
                tau_tilde = 1 / (tau_eps * bias[k] + tau0)
                mu_tilde = tau_eps * np.sum(ind * (bias_sparse - vec_combine(temp)), axis=idx) * tau_tilde
                vector[k] = np.random.normal(mu_tilde, np.sqrt(tau_tilde))
            return vector

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

        sparse_tensor = sparse_tensor_ori.copy()
        dim = np.array(sparse_tensor.shape)
        rank = 50
        vector = []
        factor = []
        for k in range(len(dim)):
            vector.append(0.1 * np.random.randn(dim[k], ))
            factor.append(0.1 * np.random.randn(dim[k], rank))
        dim = np.array(sparse_tensor.shape)
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
        for it in range(burn_iter + gibbs_iter):
            temp = sparse_tensor - temp
            mu_glb = sample_global_mu(temp[pos_obs] - vec_combine(vector)[pos_obs], pos_obs, tau_eps)
            vector = sample_bias_vector(temp - mu_glb, factor, bias, ind, dim, k, tau_eps)
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

    def BPMF(self,sparse_mat_ori, rank=50, burn_iter=1000, gibbs_iter=200):
        """Bayesian Probabilistic Matrix Factorization, BPMF."""
        import numpy as np
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

        def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0):
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
                    pos0 = np.where(sparse_mat[i, :] != 0)
                    Xt = X[pos0[0], :]
                    var_mu = tau * Xt.T @ sparse_mat[i, pos0[0]] + var_Lambda_hyper @ var_mu_hyper
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

        sparse_mat = sparse_mat_ori.copy()
        dim1, dim2 = sparse_mat.shape
        W = 0.1 * np.random.randn(dim1, rank)
        X = 0.1 * np.random.randn(dim2, rank)
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
        for it in range(burn_iter + gibbs_iter):
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

    def BTMF(self,sparse_mat_ori, rank=50, time_lags=(1, 2), burn_iter=1000, gibbs_iter=200):
        """Bayesian Temporal Matrix Factorization, BTMF."""
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
        def mvnrnd_pre(mu, Lambda):
            src = normrnd(size=(mu.shape[0],))
            return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
                            src, lower=False, check_finite=False, overwrite_b=True) + mu

        def cov_mat(mat, mat_bar):
            mat = mat - mat_bar
            return mat.T @ mat

        def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0):
            """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""

            dim1, rank = W.shape
            W_bar = np.mean(W, axis=0)
            temp = dim1 / (dim1 + beta0)
            var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
            var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_W_hyper)
            var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)

            if dim1 * rank ** 2 > 1e+8:
                vargin = 1

            if vargin == 0:
                var1 = X.T
                var2 = kr_prod(var1, var1)
                var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
                var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
                for i in range(dim1):
                    W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
            elif vargin == 1:
                for i in range(dim1):
                    pos0 = np.where(sparse_mat[i, :] != 0)
                    Xt = X[pos0[0], :]
                    var_mu = tau[i] * Xt.T @ sparse_mat[i, pos0[0]] + var_Lambda_hyper @ var_mu_hyper
                    var_Lambda = tau[i] * Xt.T @ Xt + var_Lambda_hyper
                    W[i, :] = mvnrnd_pre(solve(var_Lambda, var_mu), var_Lambda)

            return W

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

        def sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, Lambda_x):
            """Sampling T-by-R factor matrix X."""

            dim2, rank = X.shape
            tmax = np.max(time_lags)
            tmin = np.min(time_lags)
            d = time_lags.shape[0]
            A0 = np.dstack([A] * d)
            for k in range(d):
                A0[k * rank: (k + 1) * rank, :, k] = 0
            mat0 = Lambda_x @ A.T
            mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
            mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))

            var1 = W.T
            var2 = kr_prod(var1, var1)
            var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + Lambda_x[:, :, None]
            var4 = var1 @ tau_sparse_mat
            for t in range(dim2):
                Mt = np.zeros((rank, rank))
                Nt = np.zeros(rank)
                Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
                index = list(range(0, d))
                if t >= dim2 - tmax and t < dim2 - tmin:
                    index = list(np.where(t + time_lags < dim2))[0]
                elif t < tmax:
                    Qt = np.zeros(rank)
                    index = list(np.where(t + time_lags >= tmax))[0]
                if t < dim2 - tmin:
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

        def sample_precision_tau(sparse_mat, mat_hat, ind):
            var_alpha = 1e-6 + 0.5 * np.sum(ind, axis=1)
            var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind, axis=1)
            return np.random.gamma(var_alpha, 1 / var_beta)

        sparse_mat = sparse_mat_ori.copy()
        time_lags = np.array(time_lags)
        dim1, dim2 = sparse_mat.shape
        W = 0.1 * np.random.randn(dim1, rank)
        X = 0.1 * np.random.randn(dim2, rank)
        if np.isnan(sparse_mat).any() == False:
            ind = sparse_mat != 0
        elif np.isnan(sparse_mat).any() == True:
            ind = ~np.isnan(sparse_mat)
            sparse_mat[np.isnan(sparse_mat)] = 0
        tau = np.ones(dim1)
        mat_hat_plus = np.zeros((dim1, dim2))
        for it in range(burn_iter + gibbs_iter):
            tau_ind = tau[:, None] * ind
            tau_sparse_mat = tau[:, None] * sparse_mat
            W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
            A, Sigma = sample_var_coefficient(X, time_lags)
            X = sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, inv(Sigma))
            mat_hat = W @ X.T
            tau = sample_precision_tau(sparse_mat, mat_hat, ind)
            if it + 1 > burn_iter:
                mat_hat_plus += mat_hat
        mat_hat = mat_hat_plus / gibbs_iter
        mat_hat[mat_hat < 0] = 0

        return mat_hat

    def BTRMF(self,sparse_mat_ori, rank=50, time_lags=(1, 2), burn_iter=1000, gibbs_iter=200):
        """Bayesian Temporal Regularized Matrix Factorization, BTRMF."""
        import numpy as np
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

        def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0):
            """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""

            dim1, rank = W.shape
            W_bar = np.mean(W, axis=0)
            temp = dim1 / (dim1 + beta0)
            var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
            var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_W_hyper)
            var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)

            if dim1 * rank ** 2 > 1e+8:
                vargin = 1

            if vargin == 0:
                var1 = X.T
                var2 = kr_prod(var1, var1)
                var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
                var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
                for i in range(dim1):
                    W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
            elif vargin == 1:
                for i in range(dim1):
                    pos0 = np.where(sparse_mat[i, :] != 0)
                    Xt = X[pos0[0], :]
                    var_mu = tau[i] * Xt.T @ sparse_mat[i, pos0[0]] + var_Lambda_hyper @ var_mu_hyper
                    var_Lambda = tau[i] * Xt.T @ Xt + var_Lambda_hyper
                    W[i, :] = mvnrnd_pre(solve(var_Lambda, var_mu), var_Lambda)

            return W

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

        def sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, theta, Lambda_x):
            """Sampling T-by-R factor matrix X."""

            dim2, rank = X.shape
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

            var1 = W.T
            var2 = kr_prod(var1, var1)
            var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + Lambda_x[:, :, None]
            var4 = var1 @ tau_sparse_mat
            for t in range(dim2):
                Mt = np.zeros((rank, rank))
                Nt = np.zeros(rank)
                Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
                index = list(range(0, d))
                if t >= dim2 - tmax and t < dim2 - tmin:
                    index = list(np.where(t + time_lags < dim2))[0]
                elif t < tmax:
                    Qt = np.zeros(rank)
                    index = list(np.where(t + time_lags >= tmax))[0]
                if t < dim2 - tmin:
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

        def sample_precision_tau(sparse_mat, mat_hat, ind):
            var_alpha = 1e-6 + 0.5 * np.sum(ind, axis=1)
            var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind, axis=1)
            return np.random.gamma(var_alpha, 1 / var_beta)

        sparse_mat = sparse_mat_ori.copy()
        dim1, dim2 = sparse_mat.shape
        time_lags = np.array(time_lags)
        d = time_lags.shape[0]
        W = 0.1 * np.random.randn(dim1, rank)
        X = 0.1 * np.random.randn(dim2, rank)
        theta = 0.01 * np.random.randn(d, rank)
        if np.isnan(sparse_mat).any() == True:
            sparse_mat[np.isnan(sparse_mat)] = 0
        ind = sparse_mat != 0
        tau = np.ones(dim1)
        mat_hat_plus = np.zeros((dim1, dim2))
        for it in range(burn_iter + gibbs_iter):
            tau_ind = tau[:, None] * ind
            tau_sparse_mat = tau[:, None] * sparse_mat
            W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0)
            Lambda_x = sample_Lambda_x(X, theta, time_lags)
            theta = sample_theta(X, theta, Lambda_x, time_lags)
            X = sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, theta, Lambda_x)
            mat_hat = W @ X.T
            tau = sample_precision_tau(sparse_mat, mat_hat, ind)
            if it + 1 > burn_iter:
                mat_hat_plus += mat_hat
        mat_hat = mat_hat_plus / gibbs_iter
        mat_hat[mat_hat < 0] = 0

        return mat_hat

    def TRMF(self,sparse_mat, lambda_w=500,
             lambda_x=500,
             lambda_theta=500,
             eta=0.03, time_lags=(1, 2), maxiter=200):
        """Temporal Regularized Matrix Factorization, TRMF.
        还有问题，测试不能通过。
        """
        import numpy as np
        from numpy.linalg import inv as inv
        ## Initialize parameters
        rank = 50
        time_lags = np.array(time_lags)
        d = time_lags.shape[0]
        dim1, dim2 = sparse_mat.shape
        W = 0.1 * np.random.rand(dim1, rank)
        X = 0.1 * np.random.rand(dim2, rank)
        theta = 0.1 * np.random.rand(d, rank)
        ## Set hyperparameters

        pos_train = np.where(sparse_mat != 0)

        # pos_train = np.where(~np.isnan(sparse_mat))
        binary_mat = sparse_mat.copy()
        binary_mat[pos_train] = 1
        d, rank = theta.shape
        mat_hat = 0
        for it in range(maxiter):
            ## Update spatial matrix W
            for i in range(dim1):
                pos0 = np.where(sparse_mat[i, :] != 0)
                # pos0 = np.where(~np.isnan(sparse_mat[i, :]))

                Xt = X[pos0[0], :]
                vec0 = Xt.T @ sparse_mat[i, pos0[0]]
                mat0 = inv(Xt.T @ Xt + lambda_w * np.eye(rank))
                W[i, :] = mat0 @ vec0
            ## Update temporal matrix X
            for t in range(dim2):
                pos0 = np.where(sparse_mat[:, t] != 0)
                # pos0 = np.where(~np.isnan(sparse_mat[:, t]))

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

    def HaLRTC_imputer(self,sparse_tensor, alpha: list = (1 / 3, 1 / 3, 1 / 3), rho: float = 1e-5, epsilon: float = 1e-4,
                       maxiter: int = 200):
        '''
        还有问题，不能使用
        @param sparse_tensor:
        @param alpha:
        @param rho:
        @param epsilon:
        @param maxiter:
        @return:
        '''
        import numpy as np

        def ten2mat(tensor, mode):
            return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

        def mat2ten(mat, dim, mode):
            index = list()
            index.append(mode)
            for i in range(dim.shape[0]):
                if i != mode:
                    index.append(i)
            return np.moveaxis(np.reshape(mat, list(dim[index]), order='F'), 0, mode)

        def svt(mat, tau):
            u, s, v = np.linalg.svd(mat, full_matrices=False)
            vec = s - tau
            vec[vec < 0] = 0
            return np.matmul(np.matmul(u, np.diag(vec)), v)

        dim = np.array(sparse_tensor.shape)
        if not np.isnan(sparse_tensor).any():
            pos_miss = np.where(sparse_tensor == 0)
        elif np.isnan(sparse_tensor).any():
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

    def LRTC(self,sparse_tensor,
             alpha=3,
             rho=1e-5,
             theta=0.30,
             epsilon=1e-4,
             maxiter=200):
        """Low-Rank Tenor Completion with Truncated Nuclear Norm, LRTC-TNN.
        还是有问题
        """
        import numpy as np
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

        def svt_tnn(mat, alpha, rho, theta):
            tau = alpha / rho
            [m, n] = mat.shape
            if 2 * m < n:
                u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=0)
                s = np.sqrt(s)
                idx = np.sum(s > tau)
                mid = np.zeros(idx)
                mid[:theta] = 1
                mid[theta:idx] = (s[theta:idx] - tau) / s[theta:idx]
                return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
            elif m > 2 * n:
                return svt_tnn(mat.T, tau, theta).T
            u, s, v = np.linalg.svd(mat, full_matrices=0)
            idx = np.sum(s > tau)
            vec = s[:idx].copy()
            vec[theta:idx] = s[theta:idx] - tau
            return u[:, :idx] @ np.diag(vec) @ v[:idx, :]

        alpha = np.ones(alpha) / 3
        dim = np.array(sparse_tensor.shape)
        pos_missing = np.where(sparse_tensor == 0)

        X = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{X}}
        T = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{T}}
        Z = sparse_tensor.copy()
        last_tensor = sparse_tensor.copy()
        snorm = np.sqrt(np.sum(sparse_tensor ** 2))
        it = 0
        while True:
            rho = min(rho * 1.05, 1e5)
            for k in range(len(dim)):
                X[k] = mat2ten(svt_tnn(ten2mat(Z - T[k] / rho, k), alpha[k], rho, np.int(np.ceil(theta * dim[k]))), dim,
                               k)
            Z[pos_missing] = np.mean(X + T / rho, axis=0)[pos_missing]
            T = T + rho * (X - np.broadcast_to(Z, np.insert(dim, 0, len(dim))))
            tensor_hat = np.einsum('k, kmnt -> mnt', alpha, X)
            tol = np.sqrt(np.sum((tensor_hat - last_tensor) ** 2)) / snorm
            last_tensor = tensor_hat.copy()
            it += 1
            if (tol < epsilon) or (it >= maxiter):
                break
        return tensor_hat

    def CP_ALS(self,sparse_tensor_ori, rank=50, maxiter=1000):
        import numpy as np
        from numpy.linalg import inv as inv
        def kr_prod(a, b):
            return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

        def cp_combine(U, V, X):
            return np.einsum('is, js, ts -> ijt', U, V, X)

        def ten2mat(tensor, mode):
            return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

        sparse_tensor = sparse_tensor_ori.copy()
        dim1, dim2, dim3 = sparse_tensor.shape
        dim = np.array([dim1, dim2, dim3])

        U = 0.1 * np.random.rand(dim1, rank)
        V = 0.1 * np.random.rand(dim2, rank)
        X = 0.1 * np.random.rand(dim3, rank)

        if np.isnan(sparse_tensor).any():
            sparse_tensor[np.isnan(sparse_tensor)] = 0
        pos = np.where(sparse_tensor != 0)
        # pos = np.where(~np.isnan(sparse_tensor))

        binary_tensor = np.zeros((dim1, dim2, dim3))
        binary_tensor[pos] = 1
        tensor_hat = np.zeros((dim1, dim2, dim3))

        for iters in range(maxiter):
            for order in range(dim.shape[0]):
                if order == 0:
                    var1 = kr_prod(X, V).T
                elif order == 1:
                    var1 = kr_prod(X, U).T
                else:
                    var1 = kr_prod(V, U).T
                var2 = kr_prod(var1, var1)
                var3 = np.matmul(var2, ten2mat(binary_tensor, order).T).reshape([rank, rank, dim[order]])
                var4 = np.matmul(var1, ten2mat(sparse_tensor, order).T)
                for i in range(dim[order]):
                    var_Lambda = var3[:, :, i]
                    inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2 + 10e-12 * np.eye(rank))
                    vec = np.matmul(inv_var_Lambda, var4[:, i])
                    if order == 0:
                        U[i, :] = vec.copy()
                    elif order == 1:
                        V[i, :] = vec.copy()
                    else:
                        X[i, :] = vec.copy()

            tensor_hat = cp_combine(U, V, X)
        return tensor_hat
    def sarimax(self,data,order=(1,1,1),seasonal_order=(1,1,1,7),d=50):
        #勉强能用
        import statsmodels.api as sm
        import numpy as np
        mod = []
        for i in range(data.shape[0]):
            tmp = sm.tsa.statespace.SARIMAX(data[i, :d], order=order,\
                                                 seasonal_order=seasonal_order)
            res = tmp.fit()
            mod.append(res)
        for i in range(data.shape[1]):
            for j in range(data.shape[0]):
                if i <= d:
                    if np.isnan(data[j,i]):
#                     if data[j, i] == np.nan:
                        data[j, i] = (mod[j].predict(start=i, end=i ,dynamic=True))[0]
                else:
#
                    if data[j, i] == np.nan:
                        tmp = sm.tsa.statespace.SARIMAX(X[j, i - d:i], \
                                    order=order, seasonal_order=seasonal_order)
                        mod[j]=tmp.fit()
                        data[j, i] = (mod[j].predict(start=i, end=i ,dynamic=True))[0]
        return data
  
        row = data.shape[0]
        col = data.shape[1]
        g=np.ceil(col/d)
    
        for i in range(row):
            flag = 0
            count = 0
            group = 1
            for j in range(col):
                if flag==0:
                    flag = 1
                    model = sm.tsa.statespace.SARIMAX(data[i,:d],order=order,seasonal_order=seasonal_order)
                    res = model.fit()
                if count>= d:
                    count = 0
                    
                    if group<g:
                        model = sm.tsa.statespace.SARIMAX(data[i,(group-1)*d:group*d],order=order,seasonal_order=seasonal_order)
                    else:
                        model = sm.tsa.statespace.SARIMAX(data[i,(group-1)*d:],order=order,seasonal_order=seasonal_order)
                    group = group + 1
                    res = model.fit()
                if np.isnan(data[i,j]):
                    data[i,j]=(res.predict(start=j,end = j))[0]
                count = count + 1
             
 
        return data
        

