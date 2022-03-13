import numpy as np
import numpy.ma as ma
from metrics import recsys_euclidean_similarity, recsys_pearson_similarity


class UserBasedRecommendation:
    """Class that performs user-based recommendations.

    This class executes user-based recommendation based on
    some similar users rating of the items.

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
            return recsys_pearson_similarity(x, y) + 1


    def fit(self, X: np.array, miss_value: int=0):
        """Sets the ratings matrix X and miss_value attributes."""

        self.X = X
        self.miss_value = miss_value


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

        similarities = [self.metric(self.X[user_id], self.X[u]) for u in range(self.X.shape[0])]
        users = np.argsort(similarities)
        closest_users = users[users != user_id][:n_closest_users]
        user_similarities = [similarities[u] for u in closest_users]

        ratings = ma.masked_array(self.X, mask=self.X == 0, fill_value=0)
        
        closest_users_mean_ratings = ratings[closest_users].mean(axis=1)
        user_mean_rating = ratings[user_id].mean()

        predicted_ratings = []
        
        for i in range(self.X.shape[1]):
            if self.X[user_id, i] != 0:
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