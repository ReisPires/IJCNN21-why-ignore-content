import csv
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import os
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

import recresearch as rr

DATASETS_TABLE = pd.DataFrame(
    [[1,  'Anime Recommendations',     'E', 'S', '1GSz8EKsA3JlKfI-4qET0nUtmoyBGWNJl'],
     [2,  'BestBuy',                   'I', 'T', '1WZY5i6rRTBH4g8M5Qd0oSBVcWbis14Zq'],     
     [3,  'DeliciousBookmarks',        'I', 'T', '14geC9mUx1--xHkAUPtYLrMfZk4jc4ITW'],     
     [4,  'Last.FM - Listened',        'I', 'S', '1g3j9UP2a0gvB0fYJ9OzPAW1k1g59JobH'],
     [5,  'MovieLens',                 'E', 'T', '1Tbi5EVs7BBZmnuKaFHDZelFgDuz-9YEP']],
    columns=[rr.DATASET_ID, rr.DATASET_NAME, rr.DATASET_TYPE, rr.DATASET_TEMPORAL_BEHAVIOUR, rr.DATASET_GDRIVE]
)

class SparseRepr(object):
    def __init__(self, df):
        users = df[rr.COLUMN_USER_ID].unique()
        items = df[rr.COLUMN_ITEM_ID].unique()
        self._create_encoders(users, items)

    def _create_encoders(self, users, items):
        self._user_encoder = LabelEncoder()
        self._item_encoder = LabelEncoder()
        self._user_encoder.fit(users)
        self._item_encoder.fit(items)

    def get_matrix(self, users, items, interactions=None):        
        users_coo, items_coo = self._user_encoder.transform(users), self._item_encoder.transform(items)        
        data = interactions if interactions is not None else np.ones(len(users_coo))        
        sparse_matrix = sparse.coo_matrix((data, (users_coo, items_coo)), shape=(len(self._user_encoder.classes_), len(self._item_encoder.classes_))).tocsr()
        return sparse_matrix

    def get_n_users(self):
        return len(self._user_encoder.classes_)

    def get_n_items(self):
        return len(self._item_encoder.classes_)

    def get_idx_of_user(self, user):
        return self._user_encoder.transform(user) if type(user) in [list, np.ndarray] else self._user_encoder.transform([user])[0]

    def get_idx_of_item(self, item):
        return self._item_encoder.transform(item) if type(item)in [list, np.ndarray] else self._item_encoder.transform([item])[0]

    def get_user_of_idx(self, idx):
        return self._user_encoder.inverse_transform(idx) if type(idx) in [list, np.ndarray, pd.Series] else self._user_encoder.inverse_transform([idx])[0]

    def get_item_of_idx(self, idx):
        return self._item_encoder.inverse_transform(idx) if type(idx) in [list, np.ndarray, pd.Series] else self._item_encoder.inverse_transform([idx])[0]


class Dataset(object):

    def __init__(self, name, path):
        self.name = name
        self.df = self._format_df(
            pd.read_csv(path, delimiter=rr.DELIMITER, encoding=rr.ENCODING, quoting=rr.QUOTING, quotechar=rr.QUOTECHAR),           
        ).drop_duplicates(subset=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID], keep='last')
        self.sparse_repr = None

    def _format_df(self, df):        
        columns = [rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]                
        if self.name == 'Anime Recommendations':
            df = df[df[rr.COLUMN_INTERACTION]==-1]
        elif self.name == 'Book-Crossing':
            df = df[df[rr.COLUMN_INTERACTION]==0]        
        df = df[columns].dropna()        
        df = df.drop_duplicates(subset=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
        return df        
    
    def get_name(self):
        return self.name

    def get_dataframe(self):
        return self.df

    def get_n_users(self):
        return len(self.df[rr.COLUMN_USER_ID].unique())

    def get_n_items(self):
        return len(self.df[rr.COLUMN_ITEM_ID].unique())

    def get_n_interactions(self):
        return len(self.df)

    def get_min_max_rating(self):
        return (self.min_rating, self.max_rating) if self.ds_type == 'E' else None


def download_datasets(ds_dir='datasets', datasets=None, verbose=True):
    downloaded_datasets = list()
    d = 0
    for _, row in DATASETS_TABLE.iterrows():
        cur_dataset = row[rr.DATASET_NAME]
        if datasets is None or cur_dataset in datasets:
            d += 1
            if verbose:
                print('Downloading {}... ({}/{})'.format(cur_dataset, d, len(datasets) if datasets is not None else len(DATASETS_TABLE)))
            gdd.download_file_from_google_drive(file_id=row[rr.DATASET_GDRIVE], dest_path='./{}/{}.zip'.format(ds_dir, cur_dataset), unzip=True)
            os.remove('./{}/{}.zip'.format(ds_dir, cur_dataset))
            downloaded_datasets.append(cur_dataset)
    if verbose:
        print('Datasets downloaded!')
    return downloaded_datasets


def get_datasets(ds_dir='datasets', datasets=None):    
    if datasets is None:
        datasets = DATASETS_TABLE        
    if type(datasets) == pd.DataFrame:
        datasets = datasets[rr.DATASET_NAME].values    
    for ds_name in datasets:
        ds_path = os.path.join(ds_dir, ds_name, rr.FILE_INTERACTIONS)
        if not os.path.exists(ds_path):
            raise Exception("File '{}' does not exist".format(ds_path))
        yield Dataset(ds_name, ds_path)