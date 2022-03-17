import numpy as np
import random
from metrics import recsys_euclidean_similarity, recsys_pearson_similarity


class TrivialRecommendation:
    """Class that performs trivial recommendations.

    This class executes relatively simple recommendations
    that takes into account similarities.

    Attributes:
        X: matrix that represents all known ratings.
        miss_value: value that indicates the lack of rating.
        metric_name: name of metric that are used to
            compute similarities between items.

    """

    def __init__(self, metric: str='euclidean'):
        """Inits attributes to empty values and sets metric name."""

        self.X = None
        self.miss_value = None
        self.metric_name = metric


    def metric(self, x: np.array, y: np.array):
        """Metric that are used to compute similarities."""

        if self.metric_name == 'euclidean':
            return recsys_euclidean_similarity(x, y)
        elif self.metric_name == 'pearson':
            return recsys_pearson_similarity(x, y)
    

    def fit(self, X: np.array, miss_value: int=0):
        """Sets the ratings matrix X and miss_value attributes."""

        self.X = X
        self.miss_value = miss_value


    def make_recommendation(self, user_id: int, n_recommendations: int=5):
        """Makes recommendation for given user.

        This method makes recommendation by selecting the most
        similar items that were rated by collaborative users
        to some item that user highly rated.

        Args:
            user_id: id of user to recommend.
            n_recommendations: number of recommendations.

        Returns:
            list of recommended items.

        """

        highly_rated = self.X[user_id] == self.X[user_id].max()
        liked_item = random.choice(np.where(highly_rated)[0])

        collab_users = np.where(self.X[:, liked_item] != self.miss_value)[0]
        collab_items = np.unique(np.where(self.X[collab_users] != 0)[1]).tolist()

        recommendations = [i for i in collab_items
                             if self.X[user_id, i] == self.miss_value]

        recommendations.sort(key=lambda x: -self.metric(self.X[:, x],
                                                     self.X[:, liked_item]))

        return recommendations[:n_recommendations]