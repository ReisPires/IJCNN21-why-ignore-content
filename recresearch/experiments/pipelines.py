import numpy as np
from sklearn.model_selection import ParameterGrid
import time

import recresearch as rr
from recresearch.evaluation.metrics import precision_recall_score, ndcg_score, mae_score, rmse_score
from recresearch.experiments.results import get_past_results, update_past_results
from recresearch.methods import EMBEDDINGS_RECOMMENDERS
from recresearch.parameters.best_params import get_best_params, update_best_params

# ===================== Base pipeline =====================
class Pipeline(object):
    def __init__(self, model_info, dataset_name, logger=None):
        self.model_class = model_info[rr.MODEL_CLASS]
        self.embedding_class = model_info[rr.MODEL_EMBEDDING_GENERATOR]
        self.model_name = model_info[rr.MODEL_NAME]
        self.dataset_name = dataset_name        
        self.logger = logger        
        self.rec_model = None

    def _fit(self, model_params, df_train):
        return {rr.TIME_FIT: 0.0}

    def _predict(self, df_test):
        return None, {rr.TIME_PREDICT: 0.0}

    def _evaluate(self, pred, df_test):
        return dict()

    def _log(self, model_params, scores, elapsed_time):
        pass

    def _recomend(self, model_params, df_train, df_test):
        scores, elapsed_time = get_past_results(self.dataset_name, self.model_name, model_params)
        if scores is None:
            fit_time = self._fit(model_params, df_train)
            pred, predict_time = self._predict(df_test)
            scores = self._evaluate(pred, df_test)
            elapsed_time = {**fit_time, **predict_time}
            update_past_results(self.dataset_name, self.model_name, model_params, scores, elapsed_time)
        self._log(model_params, scores, elapsed_time)
        return scores


# ============== Grid Search pipeline ==============
class GridSearchPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, logger):
        super().__init__(model_info, dataset_name, logger)        
        self.grid_search_params = model_info[rr.MODEL_GRID_SEARCH_PARAMS]
    
    def _parameter_iterator(self):
        yield None

    def _get_grid_search_score(self, scores):
        return 0.0

    def run(self, df_train, df_test):
        best_score = -np.inf        
        for model_params in self._parameter_iterator():
            scores = self._recomend(model_params, df_train, df_test)
            grid_search_score = self._get_grid_search_score(scores)
            if best_score < grid_search_score:
                best_score = grid_search_score
                update_best_params(self.dataset_name, self.model_name, model_params)


# ========= Implicit recommendation pipeline ==========
class ImplicitPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, logger):
        super().__init__(model_info, dataset_name, logger)

    def _predict(self, df_test):
        start = time.time()
        pred = self.rec_model.predict(df_test, top_n=max(rr.TOP_N_VALUES))
        end = time.time()
        elapsed_time = {rr.TIME_PREDICT: end-start}
        return pred, elapsed_time

    def _evaluate(self, pred, df_test):
        prec_value, rec_value = precision_recall_score(df_test, pred, top_n=rr.TOP_N_VALUES)
        ndcg_value = ndcg_score(df_test, pred, top_n=rr.TOP_N_VALUES)
        scores = {rr.SCORE_PRECISION: prec_value, rr.SCORE_RECALL: rec_value, rr.SCORE_NDCG: ndcg_value}
        return scores

    def _log(self, model_params, scores, elapsed_time):
        self.logger.log(
            dataset=self.dataset_name,
            model=self.model_name,
            params=str(model_params),
            prec=str(scores[rr.SCORE_PRECISION]),
            rec=str(scores[rr.SCORE_RECALL]),
            ndcg=str(scores[rr.SCORE_NDCG]),
            time=str(elapsed_time)
        )

    def _get_grid_search_score(self, scores):
        return scores[rr.SCORE_NDCG][rr.TOP_N_GRID_SEARCH]


# =============== Traditional recommenders pipeline ===============
class RecommenderPipeline(Pipeline):
    def _fit(self, model_params, df_train):
        start = time.time()
        self.rec_model = self.model_class(**model_params)
        self.rec_model.fit(df_train)
        end = time.time()
        elapsed_time = {rr.TIME_FIT: end-start}
        return elapsed_time


# ---------- Grid Search ----------
class RecommenderGridSearchPipeline(RecommenderPipeline, GridSearchPipeline):
    def _parameter_iterator(self):
        params_grid = ParameterGrid(self.grid_search_params)
        for model_params in params_grid:
            yield model_params

class RecommenderImplicitGridSearchPipeline(ImplicitPipeline, RecommenderGridSearchPipeline): 
    pass



# ================== Embedding-based models pipeline ==================
class EmbeddingsPipeline(Pipeline):
    def _get_embeddings_filepath(self, embeddings_params, final_embeddings=False):
        embeddings_dir = rr.DIR_EMBEDDINGS_GRID_SEARCH if not final_embeddings else rr.DIR_EMBEDDINGS_FINAL_EXPERIMENT
        params_repr = '_'.join(['[{}]{}'.format(k, v) for k, v in sorted(embeddings_params.items())])
        if not final_embeddings:
            embeddings_filename = '_'.join([self.dataset_name, self.model_name, params_repr])        
        else:
            embeddings_filename = '_'.join([self.dataset_name, self.model_name])        
        return embeddings_dir, embeddings_filename

    def _generate_embeddings(self, embeddings_dir, embeddings_filename, embeddings_params, df_train):        
        emb_model = self.embedding_class()
        emb_model.generate_embeddings(
            df=df_train,
            embeddings_dir=embeddings_dir,
            embeddings_filename=embeddings_filename,
            **embeddings_params
        )

    def _fit(self, model_params, df_train, final_embeddings=False):
        elapsed_time = {rr.TIME_FIT: dict()}
        # Generate embeddings
        embeddings_params = model_params[rr.EMB_PARAMS_EMBEDDINGS] if rr.EMB_PARAMS_EMBEDDINGS in model_params else model_params
        embeddings_dir, embeddings_filename = self._get_embeddings_filepath(embeddings_params, final_embeddings)
        start = time.time()
        self._generate_embeddings(embeddings_dir, embeddings_filename, embeddings_params, df_train)
        end = time.time()
        elapsed_time[rr.TIME_FIT][rr.TIME_EMBEDDINGS] = end-start
        if final_embeddings:
            return elapsed_time
        # Fit recommender        
        recommender_name = model_params[rr.EMB_PARAMS_RECOMMENDERS][rr.EMB_PARAMS_REC_NAME]
        recommender_params = model_params[rr.EMB_PARAMS_RECOMMENDERS][rr.EMB_PARAMS_REC_PARAMS]
        recommender_class = EMBEDDINGS_RECOMMENDERS[recommender_name]
        start = time.time()
        self.rec_model = recommender_class(
            embeddings_dir=embeddings_dir,
            embeddings_filename=embeddings_filename,
            **recommender_params
        )
        self.rec_model.fit(df_train)
        end = time.time()
        elapsed_time[rr.TIME_FIT][rr.TIME_RECOMMENDERS] = end-start
        return elapsed_time



# ---------- Grid Search ----------
class EmbeddingsGridSearchPipeline(EmbeddingsPipeline, GridSearchPipeline):
    def _parameter_iterator(self):
        grid_search_embeddings_params = self.grid_search_params[rr.EMB_PARAMS_EMBEDDINGS]
        grid_search_recommenders = self.grid_search_params[rr.EMB_PARAMS_RECOMMENDERS]
        embeddings_params_grid = ParameterGrid(grid_search_embeddings_params)
        for embeddings_params in embeddings_params_grid:
            for recommender_name, grid_search_recommender_params in grid_search_recommenders.items():
                recommender_params_grid = ParameterGrid(grid_search_recommender_params)
                for recommender_params in recommender_params_grid:
                    model_params = {
                        rr.EMB_PARAMS_EMBEDDINGS: embeddings_params,
                        rr.EMB_PARAMS_RECOMMENDERS: {
                            rr.EMB_PARAMS_REC_NAME: recommender_name, 
                            rr.EMB_PARAMS_REC_PARAMS: recommender_params
                        }
                    }
                    yield model_params

class EmbeddingsImplicitGridSearchPipeline(ImplicitPipeline, EmbeddingsGridSearchPipeline): 
    pass



# ================== Selects appropriate pipeline ==================
def run_pipeline(model, dataset_name, logger, df_train, df_test=None):
    # Recover pipeline type
    pipeline_type = model[rr.MODEL_PIPELINE_TYPE]
    
    # Recover the correct pipeline
    if pipeline_type == rr.PIPELINE_RECOMMENDER:
        pipeline = RecommenderImplicitGridSearchPipeline(model, dataset_name, logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS:
        pipeline = EmbeddingsImplicitGridSearchPipeline(model, dataset_name, logger)
    else:
        raise Exception('Invalid recommendation or pipeline type!')

    # Execute experiments
    pipeline.run(df_train, df_test)
