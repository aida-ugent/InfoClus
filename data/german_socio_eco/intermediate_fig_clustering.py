import sys
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../src/')
from caching import from_cache

# --------------------------------Exclus clustering ploting------------------------------
adata = ad.read_h5ad(f'C:/Users/Administrator/trace/data/german_socio_eco/german_socio_eco.h5ad')
embedding = adata.obsm.get('tSNE_1')

exclus_info = from_cache('german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0')
clustering = exclus_info["clustering"]

plt.figure(figsize=(8, 6))
unique_classes = np.unique(clustering)

colors = ['red', 'green', 'blue']
for cls in unique_classes:
    indices = np.where(clustering == cls)
    plt.scatter(embedding[indices, 0], embedding[indices, 1], label=f'Cluster {cls}', alpha=0.7, color = colors[cls], s=7)

plt.legend(ncol=len(unique_classes), fontsize = 'x-large', markerscale=2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().xaxis.set_ticks([])
plt.gca().yaxis.set_ticks([])

plt.savefig('intermediate_fig_clustering.pdf')
