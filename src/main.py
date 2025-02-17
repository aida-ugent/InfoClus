import os
import pickle
import sys
import subprocess
import numpy as np

from infoclus import InfoClus
import matplotlib.pyplot as plt
import seaborn as sns
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

dataset_name = 'cytometry_2500'
# dataset_name = 'german_socio_eco'
# dataset_name = 'mushroom_3000'
embedding_name = 'tsne'

infoclus_object_path = os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}_{embedding_name}.pkl')
if_exists = os.path.exists(infoclus_object_path)
if if_exists:
    with open(os.path.join(infoclus_object_path), 'rb') as f:
        infoclus = pickle.load(f)
else:
    infoclus = InfoClus(dataset_name=dataset_name, main_emb=embedding_name)

alpha = 700
beta = 1.4
min_att=2
max_att=5
runtime_id=3
infoclus.optimise(alpha=alpha,beta=beta,min_att=min_att,max_att=max_att,runtime_id=runtime_id)

labels = infoclus._clustering_opt
tsne = infoclus.embedding
unique_classes = np.unique(labels)
num_classes = len(unique_classes)
colors = sns.color_palette("colorblind", num_classes)  # HUSL generates distinguishable colors
plt.figure(figsize=(8, 6))
for i, cls in enumerate(unique_classes):
    # Select points corresponding to the current class
    class_points = tsne[labels == cls]
    lable = f'cluster {cls}'
    plt.scatter(class_points[:, 0], class_points[:, 1],
                color=colors[i], label=lable, s=15)
plt.legend()
plt.title("Clustering of Cytometry 2500 - computed by Infoclus")
fig_path = f"../figs/Cytometry 2500 {alpha}-Infoclus"
plt.savefig(f'{fig_path}.pdf')
plt.show()





