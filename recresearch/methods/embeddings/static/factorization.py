import implicit
import os
import pickle

from recresearch.dataset import SparseRepr
import recresearch as rr

class ALSEmbeddings(object):
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, n_factors=100, n_epochs=100, regularization=0.01, verbose=True):
        model = implicit.als.AlternatingLeastSquares(
            factors=n_factors,
            regularization=regularization,
            use_gpu=False,
            iterations=n_epochs
        )   
        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID].values, df[rr.COLUMN_ITEM_ID].values)
        model.fit(sparse_matrix.T)

        items_embeddings = model.item_factors
        users_embeddings = model.user_factors
        
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'wb'))
        pickle.dump(users_embeddings, open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'wb'))
        pickle.dump(items_embeddings, open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'wb'))
        print('Finished creation of ALS embeddings!')


class BPREmbeddings(object):
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, n_factors=100, n_epochs=100, learning_rate=0.01, regularization=0.01, verbose=True):
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=n_factors,
            learning_rate=learning_rate,
            regularization=regularization,
            use_gpu=False,
            iterations=n_epochs
        )   
        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID].values, df[rr.COLUMN_ITEM_ID].values)
        model.fit(sparse_matrix.T)

        items_embeddings = model.item_factors
        users_embeddings = model.user_factors
        
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'wb'))
        pickle.dump(users_embeddings, open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'wb'))
        pickle.dump(items_embeddings, open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'wb'))
        print('Finished creation of BPR embeddings!')