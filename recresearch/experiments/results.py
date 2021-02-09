from ast import literal_eval
import json
import numpy as np
import os

import recresearch as rr

def _read_past_results():
    json_filepath = os.path.join(rr.DIR_JSON, rr.JSON_PAST_RESULTS_GRID_SEARCH)
    if not os.path.exists(json_filepath):
        return dict()
    with open(json_filepath, 'r') as prf:
        past_results_json = json.load(prf)
    return past_results_json


def _write_past_results(past_results_json):
    os.makedirs(rr.DIR_JSON, exist_ok=True)
    json_filepath = os.path.join(rr.DIR_JSON, rr.JSON_PAST_RESULTS_GRID_SEARCH)
    with open(json_filepath, 'w') as prf:
        json.dump(past_results_json, prf, indent=4, sort_keys=True)
        prf.flush()
        os.fsync(prf.fileno())


def _dict_to_str(dict_var):
    key_value_pairs = list()
    for key, value in sorted(dict_var.items(), key=lambda x: x[0]):
        key_value_pairs.append((
            "'{}'".format(key) if type(key) == str else "{}".format(key),
            "'{}'".format(value) if type(value) == str else "{}".format(value)
        ))    
    return '{' + ', '.join(['{}: {}'.format(key, value) for key, value in key_value_pairs]) + '}'


def _str_to_dict(str_var):
    return literal_eval(str_var)


def get_past_results(dataset_name, model_name, model_params):    
    past_results_json = _read_past_results()    
    model_params = _dict_to_str(model_params)    
    if (dataset_name in past_results_json
            and model_name in past_results_json[dataset_name]
            and model_params in past_results_json[dataset_name][model_name]):
        past_results = past_results_json[dataset_name][model_name][model_params]
    else:
        return None, 0.0    
    scores = _str_to_dict(past_results['scores'])
    elapsed_time = _str_to_dict(past_results['elapsed_time'])    
    return scores, elapsed_time


def update_past_results(dataset_name, model_name, model_params, scores, elapsed_time):    
    past_results_json = _read_past_results()    
    model_params = _dict_to_str(model_params)        
    if dataset_name not in past_results_json:
        past_results_json[dataset_name] = dict()
    if model_name not in past_results_json[dataset_name]:
        past_results_json[dataset_name][model_name] = dict()    
    past_results_json[dataset_name][model_name][model_params] = {
        'scores': _dict_to_str(scores), 
        'elapsed_time': _dict_to_str(elapsed_time)
    }    
    _write_past_results(past_results_json)
