import numpy as np
def knn(data_mat, n_neighbors=5, weights='uniform',missing_values=0,
        metric='nan_euclidean', copy=True, add_indicator=False):
    """

    @param data: numpy 2d array,missing values are represented by np.nan
    @param n_neighbors: number of neighbors
    @return: numpy 2d array after imputed
    """
    # 通过测试
    data = data_mat.copy()
    from sklearn.impute import KNNImputer
    imp = KNNImputer(missing_values=missing_values,n_neighbors=n_neighbors, weights=weights,
                     metric=metric, copy=copy, add_indicator=add_indicator)
    # imp = KNNImputer(n_neighbors=5)
    mat = imp.fit_transform(data)
    return mat