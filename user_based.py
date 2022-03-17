import numpy as np
import numpy.ma as ma
from sklearn.neighbors import NearestNeighbors
from metrics import recsys_euclidean_similarity, recsys_pearson_similarity


class UserBasedRecommendation:
    """Class that performs user-based recommendations.

    This class executes user-based recommendation based on
    Nearest Neighbors algorithm.

    Attributes:
        X: matrix that represents all known ratings.
        miss_value: value that indicates the lack of rating.
        metric_name: 'euclidean' or 'pearson' - name of metric
            that are used to compute similarities.
        alpha: similarity threshold.

    """

    def __init__(self, metric: str='euclidean', alpha: float=0.8,
                       miss_value: int=0):
        """Inits attributes with appropriate values."""

        self.X = None
        self.miss_value = miss_value
        self.metric_name = metric
        self.alpha = alpha


    def metric(self, x: np.array, y: np.array):
        """Metric that are used to compute similarities."""

        if self.metric_name == 'euclidean':
            return recsys_euclidean_similarity(x, y)
        elif self.metric_name == 'pearson':
            return recsys_pearson_similarity(x, y) + 1

    def NearestNeighbors_metric(self, x: np.array, y: np.array):
        """Metric that are used in Nearest Neighbors algorithm."""

        if self.metric_name == 'euclidean':
            return 1 - recsys_euclidean_similarity(x, y)
        elif self.metric_name == 'pearson':
            return 1 - recsys_pearson_similarity(x, y)


    def fit(self, X: np.array):
        """Applies fit on given ratings matrix.

        Sets the ratings matrix X and applies Nearest Neighbors
        algorithm to given ratings matrix.

        Args:
            X: ratings matrix where X[u, i] = missed_value if
                user u didn't rate item i

        Returns:
            None

        """

        self.X = X
        self.nbrs = NearestNeighbors(metric=self.NearestNeighbors_metric).fit(X)


    def find_closest_users(self, user_id: int, n_closest_users: int):
        """Finds closest users by computed neighbors matrix."""

        rng = self.nbrs.radius_neighbors([self.X[user_id]], radius=1-self.alpha, sort_results=True)
        closest_users_ids = rng[1][0]

        return closest_users_ids[1:n_closest_users+1]


    def make_recommendation(self, user_id: int, n_recommendations: int=5,
                                  n_closest_users: int=50, return_ratings: bool=False):
        """Makes user-based recommendation for given user.
        
        This method makes recommendation by calculating ratings
        for items in consideration of user (to whom recommend)
        average rating and weighted biases of the most similar
        users average ratings.

        Args:
            user_id: id of user to recommend.
            n_recommendations: number of recommendations.
            n_closest_users: number of the most similar users
                who will be considered in recommendation.
            return_rating: whether to return ratings or not.

        Returns:
            list of recommended items.

        """

        closest_users = self.find_closest_users(user_id, n_closest_users)
        user_similarities = [self.metric(self.X[user_id], u) for u in self.X[closest_users]]

        ratings = ma.masked_array(self.X, mask=(self.X == self.miss_value), fill_value=0)
        
        closest_users_mean_ratings = ratings[closest_users].mean(axis=1)
        user_mean_rating = ratings[user_id].mean()

        predicted_ratings = []
        
        for i in range(self.X.shape[1]):
            if self.X[user_id, i] != self.miss_value:
                continue
            
            user_similarities_masked = ma.masked_array(user_similarities, mask=[ratings[closest_users, i].mask])
            
            if user_similarities_masked.mask.all():
                continue
            
            bias = ratings[closest_users, i] - closest_users_mean_ratings

            predicted_rating = (user_mean_rating +
                (bias * user_similarities_masked).sum() / user_similarities_masked.sum())
            
            predicted_ratings.append((i, predicted_rating))
            
        predicted_ratings.sort(key=lambda x: -x[1])

        if return_ratings:
            return [(x[0], x[1]) for x in predicted_ratings[:n_recommendations]]
        else:
            return [x[0] for x in predicted_ratings[:n_recommendations]]