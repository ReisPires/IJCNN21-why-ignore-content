import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import turicreate as tc

import recresearch as rr


class KNN(object):
    def __init__(self, embeddings_dir, embeddings_filename, k=64):        
        self.k = k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))        
        self.embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))

    def fit(self, df):
        n_items = self.embeddings.shape[0]
        items_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * n_items))
        nearest_neighbors = np.empty((n_items, self.k))
        nearest_sims = np.empty((n_items, self.k))
        for i in range(0, n_items, items_per_batch):
            batch_sims = cosine_similarity(self.embeddings[i:i+items_per_batch], self.embeddings)
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            nearest_neighbors[i:i+items_per_batch] = np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k]
            nearest_sims[i:i+items_per_batch] = np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k]
        sim_table = tc.SFrame({
            'id_item': self.sparse_repr.get_item_of_idx(np.repeat(np.arange(n_items), self.k).astype(int)),
            'similar': self.sparse_repr.get_item_of_idx(nearest_neighbors.flatten().astype(int)),
            'score': nearest_sims.flatten()
        })
        sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]])
        self.model = tc.recommender.item_similarity_recommender.create(
            observation_data=sframe,
            user_id=rr.COLUMN_USER_ID,
            item_id=rr.COLUMN_ITEM_ID,
            similarity_type='cosine',
            only_top_k=self.k,
            nearest_items=sim_table,
            target_memory_usage=rr.MEM_SIZE_LIMIT
        )

    def predict(self, df, top_n=10):
        recommendations = self.model.recommend(
            users=df[rr.COLUMN_USER_ID].unique(),
            k=top_n,
            exclude_known=True
        ).to_dataframe().drop(columns=['score'])
        return recommendations


class UserItemSimilarity(object):
    def __init__(self, embeddings_dir, embeddings_filename):        
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))
        self.item_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.user_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'rb'))

    def fit(self, df):
        self.df_train = df.copy()
        known_items = set(self.sparse_repr._item_encoder.classes_)
        self.df_train = self.df_train[self.df_train[rr.COLUMN_ITEM_ID].isin(known_items)]

    def predict(self, df, top_n=10):
        target_users = sorted(df[rr.COLUMN_USER_ID].unique())
        top_n_items = np.empty((len(target_users), top_n), dtype=np.int32)
        users_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * self.sparse_repr.get_n_items()))
        for u in range(0, len(target_users), users_per_batch):
            batch_users = target_users[u:u+users_per_batch]
            batch_encoder = LabelEncoder()
            batch_encoder.fit(batch_users)
            batch_users = batch_encoder.inverse_transform(np.arange(len(batch_users)))
            users_idx = self.sparse_repr.get_idx_of_user(batch_users)            
            batch_sims = cosine_similarity(self.user_embeddings[users_idx], self.item_embeddings)
            known_interactions = self.df_train[self.df_train[rr.COLUMN_USER_ID].isin(batch_users)]
            batch_sims[
                batch_encoder.transform(known_interactions[rr.COLUMN_USER_ID]), 
                self.sparse_repr.get_idx_of_item(known_interactions[rr.COLUMN_ITEM_ID].values)
            ] = -np.inf
            top_n_items[u:u+users_per_batch] = np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :top_n]
        recommendations = pd.DataFrame([], columns=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_RANK])
        recommendations[rr.COLUMN_USER_ID] = np.repeat(target_users, top_n)
        recommendations[rr.COLUMN_ITEM_ID] = self.sparse_repr.get_item_of_idx(top_n_items.flatten())
        recommendations[rr.COLUMN_RANK] = np.tile(np.arange(1, top_n+1), len(target_users))
        return recommendations
