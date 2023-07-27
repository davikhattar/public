import numpy as np
import pandas as pd
import scipy


def inverse_variance_portfolio(covariance_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculates the weights for a portfolio using the inverse variance method.

    Args:
        covariance_matrix (pd.DataFrame): The covariance matrix of the assets.

    Returns:
        pd.Series: The weights of the assets in the portfolio.
    """
    weights = 1.0 / np.diag(covariance_matrix)
    weights /= weights.sum()
    return pd.Series(weights, index=covariance_matrix.index)


def minimum_variance_portfolio(covariance_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculates the weights for a portfolio using the minimum variance method.

    Args:
        covariance_matrix (pd.DataFrame): The covariance matrix of the assets.

    Returns:
        pd.Series: The weights of the assets in the portfolio.
    """
    covariance_values = covariance_matrix.values
    objective_function = lambda weights: np.dot(weights.T, np.dot(covariance_values, weights))
    constraint_function = lambda weights: np.sum(weights) - 1.0

    number_of_assets = covariance_values.shape[0]
    initial_weights = np.ones(number_of_assets) / number_of_assets
    bounds = [(0, None)] * number_of_assets
    constraints = {"type": "eq", "fun": constraint_function}
    weights = scipy.optimize.minimize(objective_function, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = pd.Series(weights.x, covariance_matrix.index)
    return weights
    

def hierarchical_risk_parity(covariance_matrix: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    """
    Calculates the weights for a portfolio using the Hierarchical Risk Parity method.
    https://quantresearch.org/HRP.py.txt

    Args:
        covariance_matrix (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str): Method for calculating the distance between the newly formed clusters (single, complete, average, weighted, ward).

    Returns:
        pd.Series: The weights of the assets in the portfolio.
    """
    correlation_matrix = covariance_to_correlation(covariance_matrix=covariance_matrix)
    # Create a hierarchical clustering linkage matrix.
    # The linkage matrix represents the hierarchical relationship between the data points based on their distances.
    link = hrp_tree_clustering(correlation_matrix=correlation_matrix, linkage_method=linkage_method)
    # This stage reorganizes the correlation matrix, so that the largest values lie along the diagonal
    # Thus similar investments are placed together, and dissimilar are placed far away
    ordered_list_tickers = hrp_quasi_diagonalization(link=link, correlation_matrix=correlation_matrix)
    # Assign weights to assets based on HRP.
    # It iteratively partitions the assets based on the dendrogram created in the previous step and assigns w to each partition.
    hrp = hrp_recursive_bisection(ordered_list_tickers=ordered_list_tickers, covariance_matrix=covariance_matrix)
    return hrp.sort_index()


def hrp_tree_clustering(correlation_matrix: pd.DataFrame, linkage_method: str = "single"):
    distance_matrix = np.sqrt((1 - correlation_matrix) / 2.0).round(5)
    distance_matrix = scipy.spatial.distance.squareform(distance_matrix.values)
    return scipy.cluster.hierarchy.linkage(distance_matrix, method=linkage_method)


def hrp_quasi_diagonalization(link: np.ndarray, correlation_matrix: pd.DataFrame):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # sortIx.append(df0)
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    sortIx = sortIx.tolist()
    sortIx = correlation_matrix.index[sortIx].tolist()
    return sortIx


def hrp_recursive_bisection(ordered_list_tickers: list, covariance_matrix: pd.DataFrame):
    def weights_ivp(covariance_matrix):
        # Compute the inverse-variance portfolio
        ivp = 1.0 / np.diag(covariance_matrix)
        ivp /= ivp.sum()
        return ivp

    def variance_cluster(covariance_matrix, list_tickers_cluster):
        # Compute variance per cluster
        cov_ = covariance_matrix.loc[list_tickers_cluster, list_tickers_cluster]
        w_ = weights_ivp(cov_).reshape(-1, 1)
        cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return cVar

    weights = pd.Series(1, index=ordered_list_tickers)  # Starts with equal weights for all assets
    clusters = [ordered_list_tickers]  # Create one cluster with all assets (len==1)
    while len(clusters) > 0:
        _clusters = []
        for cluster in clusters:
            if len(cluster) > 1:
                half = len(cluster) // 2
                _clusters.append(cluster[:half])
                _clusters.append(cluster[half:])
        clusters = _clusters
        for c in range(0, len(clusters), 2):
            tickers1 = clusters[c]
            variance1 = variance_cluster(covariance_matrix, tickers1)
            tickers2 = clusters[c + 1]
            variance2 = variance_cluster(covariance_matrix, clusters[c + 1])
            alpha = 1 - variance1 / (variance1 + variance2)
            weights[tickers1] = weights[tickers1] * alpha
            weights[tickers2] = weights[tickers2] * (1 - alpha)
    return weights


def covariance_to_correlation(covariance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a covariance matrix to a correlation matrix.

    Args:
        covariance_matrix: The covariance matrix to be converted.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    std = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std, std)
    correlation_matrix[correlation_matrix < -1] = -1
    correlation_matrix[correlation_matrix > 1] = 1
    if not isinstance(covariance_matrix, pd.DataFrame):
        return correlation_matrix
    return pd.DataFrame(correlation_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)
