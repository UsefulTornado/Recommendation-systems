import numpy as np
import numpy.ma as ma


class ElementaryRecommendation:
    """Class that performs elementary recommendations.

    This class executes sorely simple recommendations
    that could be used as a baseline. Class is based on
    average rating of items and performs predictions
    by computing bias for each user and item.

    Attributes:
        ratings: matrix that represents all known ratings.
        predictions: matrix with both known and predicted ratings.

    """

    def __init__(self):
        """Inits all attributes of class to empty values."""

        self.ratings = None
        self.predictions = None


    def fit(self, X: np.array, miss_value: int=0, smoothing_coeff: float=10):
        """Takes matrix with ratings and computes missed values.

        This method takes ratings matrix, store it as attribute, computes
        ratings that are to be predicted and store predictions as well.

        Args:
            X: matrix (numUsers x numItems) with ratings where X[u, i] = miss_value
                if rating doesn't exist.
            miss_value: special value that represents miss of appropriate rating.
            smoothing_coeff: coefficient fot smoothing and to avoid zero division as well.

        Returns:
            None

        """

        ratings = ma.masked_array(X, mask=(X == miss_value), fill_value=miss_value)
        avg_rating = ratings.mean()

        users_bias = ((ratings - avg_rating).sum(axis=1) /
                     ((~ratings.mask).sum(axis=1) + smoothing_coeff))
        users_bias.fill_value = 0

        items_bias = ((ratings - users_bias[..., None] - avg_rating).sum(axis=0) /
                     ((~ratings.mask).sum(axis=0) + smoothing_coeff))
        items_bias.fill_value = 0

        self.ratings = ratings
        self.predictions = avg_rating + np.outer(users_bias, items_bias)


    def make_recommendation(self, user_id: int, n_recommendations: int=5, return_ratings: bool=False):
        """Makes recommendation for given user.

        This method makes recommendation by selecting the most
        suitable items depending on predicted ratings. 

        Args:
            user_id: id of user to recommend.
            n_recommendations: number of recommendations.
            return_ratings: whether to return recommendations with predicted ratings.

        Returns:
            list of recommended items.

        """

        predictions = self.predictions[user_id]
        recommendations = []

        for item_id in np.argsort(predictions)[::-1]:
            if not ma.is_masked(predictions[item_id]):
                recommendations.append((item_id, predictions[item_id]))

            if len(recommendations) == n_recommendations:
                break
                
        if return_ratings:
            return [(x[0], x[1]) for x in recommendations]
        else:
            return [x[0] for x in recommendations]