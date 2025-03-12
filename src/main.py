import os
import pickle
import sys
import subprocess

from infoclus import InfoClus
from sklearn.cluster import AgglomerativeClustering, KMeans

def get_project_root():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up the directory tree until you find a specific file (like `setup.py` or a config file)
    # In this case, we can stop when we find the root of the project (you can customize the condition)
    while not os.path.exists(os.path.join(current_dir,
                                          'readme.md')):  # You can change 'setup.py' to something else (e.g., README.md or a custom marker file)
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Stop when you reach the root of the filesystem
            raise RuntimeError("Project root not found.")
        current_dir = parent_dir

    return current_dir
ROOT_DIR = get_project_root()
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

