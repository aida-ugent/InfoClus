import sys
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../src/')
from caching import from_cache

# --------------------------------Exclus clustering ploting------------------------------
adata = ad.read_h5ad(f'C:/Users/Administrator/trace/data/mushroom_binary/mushroom_binary.h5ad')
embedding = adata.obsm.get('tSNE_1')

exclus_info = from_cache('mushroom_binary-tSNE_1-single-800-1.5-2-30-0-0')
clustering = exclus_info["clustering"]

plt.figure(figsize=(8, 6))
unique_classes = np.unique(clustering)

for cls in unique_classes:
    indices = np.where(clustering == cls)
    plt.scatter(embedding[indices, 0], embedding[indices, 1], label=f'Cluster {cls}', s=3)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(unique_classes),  markerscale=2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().xaxis.set_ticks([])
plt.gca().yaxis.set_ticks([])

plt.savefig('intermediate_fig_clustering.pdf')
