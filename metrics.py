import numpy as np


def euclidean_distance(x: np.array, y: np.array) -> float:
    """Calculates euclidean distance.
    
    This function calculates euclidean distance between
    two points x and y in Euclidean n-space.

    Args:
        x, y: points in Euclidean n-space.

    Returns:
        length of the line segment connecting given points.

    """

    return np.sqrt(np.sum((x - y) ** 2))


def euclidean_similarity(x: np.array, y: np.array) -> float:
    """Calculates euclidean similarity.
    
    This function calculates euclidean similarity
    between points x and y in Euclidean n-space
    that based on euclidean distance between this points.

    Args:
        x, y: two points in Euclidean n-space.

    Returns:
        similarity between points x and y.

    """

    return 1 / (1 + euclidean_distance(x, y))


def pearson_similarity(x: np.array, y: np.array) -> float:
    """Calculates Pearson correlation coefficient.
    
    This function calculates Pearson correlation coefficient
    for given 1-D data arrays x and y.

    Args:
        x, y: two 1-D data arrays with float values.

    Returns:
        Pearson correlation between x and y.

    """

    x_b, y_b = x - x.mean(), y - y.mean()

    denominator = np.sqrt(np.sum(x_b ** 2) * np.sum(y_b ** 2))

    if denominator == 0:
        return np.nan

    return np.sum(x_b * y_b) / denominator


def euclidean_users_similarity(x: np.array, y: np.array) -> float:
    """Calculates euclidean similarity of two users.

    This function calculates users similarity based on
    euclidean similarity. The similarity between two users,
    that represented here as two 1-D arrays where each value
    is score for some item, is calculated only for
    those indices i for which there is score both
    for first and second user. It's assumed that zero values
    in arrays are equal to lack of score for some items.

    Args:
        x, y: two 1-D data arrays that represent user's scores.

    Returns:
        euclidean similarity between two users x and y.

    """

    x_new, y_new = x[(x != 0) & (y != 0)], y[(x != 0) & (y != 0)]
    
    if x_new.shape[0] == 0:
        return -np.inf

    return euclidean_similarity(x_new, y_new)


def pearson_users_similarity(x: np.array, y: np.array) -> float:
    """Calculates pearson similarity of two users.

    This function calculates users similarity based on
    pearson similarity. The similarity between two users,
    that represented here as two 1-D arrays where each value
    is score for some item, is calculated only for
    those indices i for which there is score both
    for first and second user. It's assumed that zero values
    in arrays are equal to lack of score for some items.

    Args:
        x, y: two 1-D data arrays that represent user's scores.

    Returns:
        pearson similarity between two users x and y.

    """
    
    x_new, y_new = x[(x != 0) & (y != 0)], y[(x != 0) & (y != 0)]

    if x_new.shape[0] == 0:
        return -np.inf
    
    return pearson_similarity(x_new, y_new)