import recresearch as rr

MF_PARAMS = {
    'n_factors': [50, 100, 300],
    'regularization': [1e-2, 1e-1, 1]
}

EMBEDDINGS_PARAMS = {
    'embedding_dim': [50, 100, 300],    
    'negative_sampling': [10, 20, 30],
    'negative_exponent': [0.5, 0.75, 1],    
}

ITEM2VEC_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'KNN': dict()}
}

USER2VEC_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'UIS': dict()}
}