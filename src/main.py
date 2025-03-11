import os
import pickle
import sys
import subprocess

from infoclus import InfoClus
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

# dataset_name = 'cytometry_2500'
dataset_name = 'german_socio_eco'
# dataset_name = 'mushroom'

embedding_name = 'tsne'

model = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None)
# model = KMeans(n_clusters=3, random_state=42)

if isinstance(model, AgglomerativeClustering):
    infoclus_object_path = os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}_{embedding_name}_agglomerative_{model.linkage}.pkl')
if isinstance(model, KMeans):
    infoclus_object_path = os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}_{embedding_name}_kmeans_{model.n_clusters}.pkl')
if_exists = os.path.exists(infoclus_object_path)

if if_exists:
    with open(os.path.join(infoclus_object_path), 'rb') as f:
        infoclus = pickle.load(f)
else:
    infoclus = InfoClus(dataset_name=dataset_name, main_emb=embedding_name, model=model)

alpha = 50
beta = 1.5
min_att=2
max_att=5
runtime_id=3
infoclus.optimise(alpha=alpha,beta=beta,min_att=min_att,max_att=max_att,runtime_id=runtime_id)

infoclus.visualize_result(show_now_embedding=True, save_embedding=True, show_now_explanation=True, save_explanation=True)

