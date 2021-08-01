# KDMiLe 2021 - Why Ignore Content?
Source code of the paper "Why Ignore Content? An Intrinsic Evaluation of Item Embeddings for Collaborative Filtering", submitted at the Symposium on Knowledge Discovery, Mining and Learning 2021

## How to reproduce experiments
To fully reproduce the experiments presented on the papers, you'll have to perform 7 steps:
1. Prepare the environment
2. Download datasets
3. Create embeddings
4. Build content matrices
5. Build similarity tables
6. Get intruder detection task data
7. Run CB-NDCG and Automatic Feature Prediction

Each step is explained below...

### 1: Prepare the environment
First, you'll have to download all packages used in the code. We strongly recommend you to create an independent **Python 3.8** environment, to ensure every package will run smoothly and avoid messing with other packages.

To install the packages, simply run `pip install -r requirements.txt`

### 2. Download datasets
Before running the experiments, you'll have to download the datasets. This can be done with the command `python3 dataset_downloader.py`

All datasets will be downloaded and stored in the folder _datasets_, which will be created.

### 3. Create embeddings
To perform parameter optimization through grid search and generate the final embeddings, you must run `python3 main.py`

The code execution will take some time, and it will create three folders: (1) _embeddings_, where all embeddings created during parameter optimization (_embeddings/grid search_), as well as the final embeddings (_embeddings/final_experiment_), will be stored; (2) _jsons_, having .json files with the results and best parameters; and (3) _logs_, with logs of the experiment.

### 4. Build content matrices
All evaluation tasks uses the content of the items in a way or another. Because of that, you'll have to create sparse matrices containing the tags or categories of the items, according to the dataset. This can be done with `python3 create_content_matrices.py`

### 5. Build similarity tables
After creating the embeddings for Last.FM and MovieLens datasets, you can generate similarity tables for different artists and movies with `python3 similarity_tables.py`

### 6. Get intruder detection task data
The data used for the intruder detection task can be created using the command `python3 intruder_detection.py`

### 7. Run CB-NDCG and Automatic Feature Prediction
Finally, you can get the scores of CB-NDCG, as well as the accuracy in the auto-tagging task, with the command `python3 cbndcg_autotagging.py`

---
Pedro R. Pires

MSc student at Federal University of São Carlos, Brazil.
