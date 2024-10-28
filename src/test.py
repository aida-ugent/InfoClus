# todo: combine infoclus result and h5 file from Edith
import anndata as ad
from os.path import join

import numpy as np

from utils import load_data
from caching import from_cache

DATA_SET_NAME = 'german_socio_eco'

# todo: understand h5 file and combine with infoclus result

#  write different data version into layer of adata, and set the final scaled one as X
adata = ad.read_h5ad(f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}/{DATA_SET_NAME}.h5ad')
DATA_FOLDER = f'C:/Users/Administrator/OneDrive - UGent/Documents/Data/ExClus/{DATA_SET_NAME}'
DATA_FILE = f'{DATA_SET_NAME}.csv'
df_data, df_data_scaled, lenBinary = load_data(join(DATA_FOLDER, DATA_FILE))

adata.layers["raw_data"] = df_data.copy().values
adata.layers["scaled_data"] = df_data_scaled.copy().values
adata.X = adata.layers["scaled_data"].copy()

# todo: put extract information from file to h5
infoclus = from_cache('C:/Users/Administrator/OneDrive - UGent/Documents/GitHub/ExClus/data/german_socio_eco/german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0')
clustering = infoclus["clustering"]
stasPrior = infoclus['prior']
featureType = infoclus['dls']
si = infoclus['si']
maxClusLabel = infoclus['maxlabel']
stasClus = infoclus['infor']
attsClus = infoclus['attributes']
icsClus = infoclus['ic']

adata.obs['infoclus_clustering'] = clustering
adata.uns['ExClus'] = {}
adata.uns['ExClus']['si'] = si
for cluster in range(maxClusLabel+1):
    # todo: decide whether to store statitics based on raw data instead of scaled data
    adata.uns['ExClus'][f'cluster_{cluster}'] = {}
    adata.uns['ExClus'][f'cluster_{cluster}']['mean'] = stasClus[cluster][0]
    adata.uns['ExClus'][f'cluster_{cluster}']['var'] = stasClus[cluster][1]
    adata.uns['ExClus'][f'cluster_{cluster}']['count'] = stasClus[cluster][2]
    adata.uns['ExClus'][f'cluster_{cluster}']['ic'] = np.array(icsClus[cluster])

adata.var['prior_mean'] = stasPrior[:,0]
adata.var['prior_var'] = stasPrior[:,1]
adata.var['feature_type'] = featureType

adata.write(f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}/{DATA_SET_NAME}.h5ad')

# import numpy as np
# import scanpy as sc
# from scipy.sparse import csr_matrix
#
# # Create a small example AnnData object with random data
# n_cells = 100
# n_genes = 200
# data = np.random.poisson(1, (n_cells, n_genes))
#
# # Convert to a sparse matrix and initialize AnnData
# adata = sc.AnnData(X=csr_matrix(data))
#
# # Example: Store raw counts
# adata.layers["raw_counts"] = adata.X.copy()
#
# # Normalize the data and store it in a separate layer
# sc.pp.normalize_total(adata, target_sum=1e4)
# adata.layers["normalized_counts"] = adata.X.copy()
#
# # Log-transform the normalized data and store it in another layer
# sc.pp.log1p(adata)
# adata.layers["log_normalized_counts"] = adata.X.copy()
#
# # Scale the data (centering and standardizing) and store in yet another layer
# sc.pp.scale(adata)
# adata.layers["scaled_counts"] = adata.X.copy()
#
# # Reset the main .X matrix if needed
# adata.X = adata.layers["raw_counts"].copy()
#
# # Save the AnnData object to an .h5ad file
# adata.write("example_data.h5ad")
#
#

