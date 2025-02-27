import os
import pickle
import sys
import subprocess
import pandas as pd
import csv

from src.infoclus import InfoClus
from sklearn.cluster import AgglomerativeClustering, KMeans

def get_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            universal_newlines=True
        ).strip()
        return root
    except subprocess.CalledProcessError:
        raise RuntimeError("This directory is not a Git repository.")
ROOT_DIR = get_root()
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

samples_size = [2500,5000,10000,20000,30000,50000,80000]
alphas = [1500,3000,6000,12000,18000,30000,48000]
beta = 1.5
min_att=2
max_att=5
runtime_ids = [3,4,5,5,5,6,6]

dataset = 'cytometry'
data_folder = os.path.join(ROOT_DIR, 'data', dataset)
embedding_name = 'tsne'
model = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None)

columns = ["initialization_time", "splitting_runtime", "splitting_count", "ave_splitting_time", "clustering_count", "ave_clustering_time"]
df = pd.DataFrame(columns=columns)
df.to_csv("../data/cytometry/scalability_output.csv", index=True, index_label='sample_size')

for i in range(len(samples_size)):

    sample = samples_size[i]
    dataset_name = f'{dataset}_{sample}'

    infoclus_object_path = ''
    if isinstance(model, AgglomerativeClustering):
        infoclus_object_path = os.path.join(data_folder, f'{dataset_name}_{embedding_name}_agglomerative_{model.linkage}.pkl')
    if isinstance(model, KMeans):
        infoclus_object_path = os.path.join(data_folder, f'{dataset_name}_{embedding_name}_kmeans_{model.n_clusters}.pkl')
    if_exists = os.path.exists(infoclus_object_path)
    if if_exists:
        with open(os.path.join(infoclus_object_path), 'rb') as f:
            infoclus = pickle.load(f)
    else:
        infoclus = InfoClus(dataset_name=dataset_name, main_emb=embedding_name, model=model, data_folder=data_folder)

    alpha = alphas[i]
    beta = beta
    runtime_id=runtime_ids[i]

    infoclus.optimise(alpha=alpha,beta=beta,min_att=min_att,max_att=max_att,runtime_id=runtime_id)
