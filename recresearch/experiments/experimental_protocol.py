from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

import recresearch as rr
from recresearch.dataset import get_datasets
from recresearch.experiments import pipelines as pp
from recresearch.experiments.logger import BasicLogger, ImplicitResultsLogger
from recresearch.methods import MODELS_TABLE
from recresearch.parameters.best_params import get_best_params
from recresearch.utils.preprocessing import recsys_sampling, cut_by_minimal_interactions, remove_cold_start


class ExperimentalProtocol(object):
    def __init__(self, args):                        
        self._set_loggers()

    # Initialize loggers
    def _set_loggers(self):        
        logger_name = '{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))        
        grid_search_logger_name = logger_name + '_gridsearch.log'        
        error_logger_name = logger_name + '_error.log'        
        self.grid_search_logger = ImplicitResultsLogger(grid_search_logger_name)        
        self.error_logger = BasicLogger(error_logger_name)

    # Run experiment
    def run(self):
        # Train-test split rules
        train_size = 0.8        
        test_size = 0.2
        
        # Iterate over datasets...
        for dataset in get_datasets(rr.DIR_DATASETS):
            
            # Get name and interactions dataframe
            dataset_name = dataset.get_name()
            df_full = dataset.get_dataframe()

            # Perform split
            df_train, df_test = train_test_split(df_full, train_size=train_size, test_size=test_size, shuffle=True, random_state=rr.RANDOM_SEED)

            # =================== GRID SEARCH ===================
            # Apply subsampling if needed
            df_train = recsys_sampling(df_train, dataset_name, rr.SAMPLING_RATE_HYPERPARAMETERIZATION)

            # Remove items or users with few interactions
            df_train = cut_by_minimal_interactions(df_train, min_interactions=2)

            # Remove cold start
            df_test = remove_cold_start(df_train, df_test)

            # Run methods
            try:
                for _, model in MODELS_TABLE.iterrows():
                    pp.run_pipeline(model, dataset_name, self.grid_search_logger, df_train, df_test)
            except Exception as e:
                self.error_logger.log('Problems during grid search for model {} and dataset {}'.format(model[rr.MODEL_NAME], dataset_name))
                self.error_logger.log(e)
                continue
            
            # Clean memory
            del df_train, df_test
            
            # ================ FINAL EMBEDDINGS ================
            # Apply subsampling if needed
            df_full = recsys_sampling(df_full, dataset_name, rr.SAMPLING_RATE_EXPERIMENT)

            # Remove items or users with few interactions
            df_full = cut_by_minimal_interactions(df_full, min_interactions=2)
            
            # Generate embeddings
            try:
                for _, model in MODELS_TABLE.iterrows():
                    emb_pp = pp.EmbeddingsPipeline(model, dataset_name)
                    emb_params = get_best_params(dataset_name, model[rr.MODEL_NAME])
                    emb_pp._fit(emb_params, df_full, final_embeddings=True)
            except Exception as e:
                self.error_logger.log('Problems while creating embeddings for model {} and dataset {}'.format(model[rr.MODEL_NAME], dataset_name))
                self.error_logger.log(e)
                continue

        print('Embeddings created!')