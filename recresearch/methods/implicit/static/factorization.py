import implicit
import pandas as pd
import turicreate as tc

import recresearch as rr
from recresearch.dataset import SparseRepr


class ALSImplicit(object):
    def __init__(self, n_factors, regularization=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.regularization = regularization
        self.n_epochs = n_epochs
        
    def fit(self, df):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors, 
            regularization=self.regularization,
            use_gpu=False, 
            iterations=self.n_epochs
        )
        self.sparse_repr = SparseRepr(df)
        self.sparse_matrix = self.sparse_repr.get_matrix(df[rr.COLUMN_USER_ID].values, df[rr.COLUMN_ITEM_ID].values)
        self.model.fit(self.sparse_matrix.T)

    def predict(self, df, top_n=10):
        users = self.sparse_repr.get_idx_of_user(df[rr.COLUMN_USER_ID].unique())
        recommendations = self.model.recommend_all(user_items=self.sparse_matrix)[users, :]
        recommendations = pd.DataFrame([
            [self.sparse_repr.get_user_of_idx(user), self.sparse_repr.get_item_of_idx(item), r]
            for user, items in zip(users, recommendations)
            for r, item in enumerate(items, start=1)
        ], columns=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_RANK])
        return recommendations


class BPRImplicit(object):
    def __init__(self, n_factors, regularization=0.01, learning_rate=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        
    def fit(self, df):
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.n_factors, 
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            use_gpu=False, 
            iterations=self.n_epochs
        )
        self.sparse_repr = SparseRepr(df)
        self.sparse_matrix = self.sparse_repr.get_matrix(df[rr.COLUMN_USER_ID].values, df[rr.COLUMN_ITEM_ID].values)
        self.model.fit(self.sparse_matrix.T)

    def predict(self, df, top_n=10):
        users = self.sparse_repr.get_idx_of_user(df[rr.COLUMN_USER_ID].unique())
        recommendations = self.model.recommend_all(user_items=self.sparse_matrix)[users, :]
        recommendations = pd.DataFrame([
            [self.sparse_repr.get_user_of_idx(user), self.sparse_repr.get_item_of_idx(item), r]
            for user, items in zip(users, recommendations)
            for r, item in enumerate(items, start=1)
        ], columns=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_RANK])
        return recommendations
