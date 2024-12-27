import os
import pickle
import sys
import subprocess
from infoclus import InfoClus
from utils import get_git_root

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
# dataset_name = 'german_socio_eco'
dataset_name = 'mushroom_3000'
embedding_name = 'tsne'

infoclus_object_path = os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}_{embedding_name}.pkl')
if_exists = os.path.exists(infoclus_object_path)
if if_exists:
    with open(os.path.join(infoclus_object_path), 'rb') as f:
        infoclus = pickle.load(f)
else:
    infoclus = InfoClus(dataset_name=dataset_name, main_emb=embedding_name)
infoclus.optimise()





