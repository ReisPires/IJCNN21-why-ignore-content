from gensim.models import Word2Vec
from multiprocessing import cpu_count
import numpy as np
import os
import pickle

import recresearch as rr
from recresearch.dataset import SparseRepr

class Item2Vec(object):
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, embedding_dim=50, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=1e-3):
        # Verifies if embeddings have already been previously created
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(item_embeddings_filepath):        
            print('Embeddings already created...')
            return
        
        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID], df[rr.COLUMN_ITEM_ID])

        print('Generating interactions file...')        
        fid = 0
        interactions_file = 'item2vec_interactions_{}.temp'.format(fid)
        while os.path.exists(interactions_file):
            fid += 1
            interactions_file = 'item2vec_interactions_{}.temp'.format(fid)        
        with open(interactions_file, 'w') as f:
            for user in range(sparse_matrix.shape[0]):
                f.write(' '.join(sparse_matrix[user].nonzero()[1].astype(str)) + '\n')
                
        print('Generating Item2Vec embeddings...')            
        model = Word2Vec(
            corpus_file=interactions_file,
            size=embedding_dim,
            window=sparse_matrix.sum(axis=1).max()*10000,
            min_count=1,
            workers=cpu_count(),
            sg=1,
            hs=0,
            negative=negative_sampling,
            ns_exponent=negative_exponent,
            sample=0 if subsampling_p is None else subsampling_p,
            max_vocab_size=None,
            max_final_vocab=None,            
            iter=n_epochs,
            trim_rule=None,
            sorted_vocab=0,            
            compute_loss=False,
            seed=rr.RANDOM_SEED
        )
        embeddings = model.wv.vectors[np.argsort(np.fromiter(model.wv.index2word, dtype=np.int32, count=len(model.wv.index2word)))]
        os.remove(interactions_file)
       
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(embeddings, open(item_embeddings_filepath, 'wb'))
        print('Finished creation of Item2Vec embeddings!')