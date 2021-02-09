import pandas as pd

import recresearch as rr

# Modelos
from recresearch.methods.implicit.static import ALSImplicit, BPRImplicit
from recresearch.methods.embeddings.static import ALSEmbeddings, BPREmbeddings
from recresearch.methods.embeddings.static import Item2Vec, User2Vec
from recresearch.methods.embeddings.static import KNN, UserItemSimilarity

# Parametros
from recresearch.parameters.grid_search import MF_PARAMS
from recresearch.parameters.grid_search import ITEM2VEC_PARAMS, USER2VEC_PARAMS

# ===================== Tabela de modelos =====================
MODELS_TABLE = pd.DataFrame(
    [[8,   'ALS',       rr.PIPELINE_RECOMMENDER,   ALSImplicit,      ALSEmbeddings,           MF_PARAMS],
     [9,   'BPR',       rr.PIPELINE_RECOMMENDER,   BPRImplicit,      BPREmbeddings,           MF_PARAMS],     
     [11,  'Item2Vec',  rr.PIPELINE_EMBEDDINGS,    Item2Vec,         Item2Vec,          ITEM2VEC_PARAMS],
     [12,  'User2Vec',  rr.PIPELINE_EMBEDDINGS,    User2Vec,         User2Vec,          USER2VEC_PARAMS]],     
    columns=[rr.MODEL_ID, rr.MODEL_NAME, rr.MODEL_PIPELINE_TYPE, rr.MODEL_CLASS, rr.MODEL_EMBEDDING_GENERATOR, rr.MODEL_GRID_SEARCH_PARAMS]
)

# ======= Dicionario de recomendadores para modelos de embeddings =======
EMBEDDINGS_RECOMMENDERS = {
    'KNN': KNN,    
    'UIS': UserItemSimilarity
}