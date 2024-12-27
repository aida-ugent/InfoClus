import os
import pickle
from src.infoclus import InfoClus
from src.frontend_dash.utils import check_infoclus_object
from utils import get_git_root

ROOT_DIR = get_git_root()

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





