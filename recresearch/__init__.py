import csv

# File names
FILE_ITEMS = 'items.csv'
FILE_USERS = 'users.csv'
FILE_INTERACTIONS = 'interactions.csv'

# CSV information
DELIMITER = ';'
QUOTECHAR = '"'
QUOTING = csv.QUOTE_ALL
ENCODING = "ISO-8859-1"

# Dirs
DIR_LOG = 'logs'
DIR_DATASETS = 'datasets'
DIR_EMBEDDINGS_GRID_SEARCH = 'embeddings/grid_search'
DIR_EMBEDDINGS_FINAL_EXPERIMENT = 'embeddings/final_experiment'

# Columns of files and dataframes
COLUMN_ITEM_ID = 'id_item'
COLUMN_ITEM_NAME = 'name_item'
COLUMN_USER_ID = 'id_user'
COLUMN_USER_NAME = 'name_user'
COLUMN_INTERACTION = 'interaction'
COLUMN_RANK = 'rank'
COLUMN_DATETIME = 'datetime'
COLUMN_TIMESTAMP = 'timestamp'

# Columns of datasets table
DATASET_ID = 'id'
DATASET_NAME = 'name'
DATASET_TYPE = 'type'
DATASET_TEMPORAL_BEHAVIOUR = 'temporal_behaviour'
DATASET_GDRIVE = 'gdrive'

# Columns of models table
MODEL_ID = 'id'
MODEL_NAME = 'name'
MODEL_RECOMMENDATION_TYPE = 'recommendation_type'
MODEL_PIPELINE_TYPE = 'pipeline_type'
MODEL_CLASS = 'model_class'
MODEL_GRID_SEARCH_PARAMS = 'params'
MODEL_EMBEDDING_GENERATOR = 'embedding_gen'

# Types of pipeline
PIPELINE_RECOMMENDER = 'R'
PIPELINE_EMBEDDINGS = 'E'

# Experiment parameters
EXPERIMENT_TYPE = 'type'
EXPERIMENT_TEMPORAL_BEHAVIOUR = 'temporal_behaviour'
EXPERIMENT_DATASETS = 'datasets'
EXPERIMENT_MODELS = 'models'
EXPERIMENT_GRID_SEARCH = 'grid_search'

# Date format
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Seed
RANDOM_SEED = 2020

# Sampling rates
SAMPLING_RATE_HYPERPARAMETERIZATION = {
    'Anime Recommendations': 0.3,    
    'MovieLens': 0.1,     
}
SAMPLING_RATE_EXPERIMENT = {
    'Anime Recommendations': 0.3,    
    'MovieLens': 0.1,        
}

# RAM max size
MEM_SIZE_LIMIT = 1.28e+9

# Embeddings files
FILE_SPARSE_REPR = '{}_sparse_repr.pkl'
FILE_ITEM_EMBEDDINGS = '{}_item_embeddings.pkl'
FILE_USER_EMBEDDINGS = '{}_user_embeddings.pkl'

# Embeddings parameters
EMB_PARAMS_EMBEDDINGS = 'embeddings'
EMB_PARAMS_RECOMMENDERS = 'recommenders'
EMB_PARAMS_REC_NAME = 'name'
EMB_PARAMS_REC_PARAMS = 'params'

# Keys for storing elapsed time
TIME_FIT = 'fit'
TIME_PREDICT = 'predict'
TIME_EMBEDDINGS = 'embeddings'
TIME_RECOMMENDERS = 'recommenders'

# Scores
SCORE_RMSE = 'rmse'
SCORE_MAE = 'mae'
SCORE_PRECISION = 'prec'
SCORE_RECALL = 'rec'
SCORE_NDCG = 'ndcg'

# JSON files
DIR_JSON = 'jsons'
JSON_BEST_PARAMS = 'best_params.json'
JSON_PAST_RESULTS_GRID_SEARCH = 'grid_search.json'
JSON_PAST_RESULTS_FINAL_EXPERIMENT = 'final_experiment.json'

# Top-N values
TOP_N_VALUES = list(range(1, 31))
TOP_N_GRID_SEARCH = 15