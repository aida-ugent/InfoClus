import anndata as ad
from os.path import join

import numpy as np

from src.utils import load_data
from src.caching import from_cache


def dicts2adata(dataset_name: str, adata_path: str, dicts_path: str):

    #  write different data version into layer of adata, and set the final scaled one as X
    adata = ad.read_h5ad(adata_path)
    DATA_FOLDER = f'C:/Users/Administrator/OneDrive - UGent/Documents/Data/ExClus/{dataset_name}'
    DATA_FILE = f'{dataset_name}.csv'
    df_data, df_data_scaled, lenBinary = load_data(join(DATA_FOLDER, DATA_FILE))

    adata.layers["raw_data"] = df_data.copy().values
    adata.layers["scaled_data"] = df_data_scaled.copy().values
    adata.X = adata.layers["scaled_data"].copy()

    infoclus = from_cache(dicts_path)
    clustering = infoclus["clustering"]
    stasPrior = infoclus['prior']
    featureType = infoclus['dls']
    si = infoclus['si']
    maxClusLabel = infoclus['maxlabel']
    stasClus = infoclus['infor']
    attsClus = infoclus['attributes']
    icsClus = infoclus['ic']

    adata.obs['infoclus_clustering'] = clustering
    adata.uns['InfoClus'] = {}
    adata.uns['InfoClus']['si'] = si
    adata.uns['InfoClus']['main_emb'] = 'tSNE_1'
    for cluster in range(maxClusLabel+1):
        # todo: decide whether to store statitics based on raw data instead of scaled data
        adata.uns['InfoClus'][f'cluster_{cluster}'] = {}
        adata.uns['InfoClus'][f'cluster_{cluster}']['mean'] = stasClus[cluster][0]
        adata.uns['InfoClus'][f'cluster_{cluster}']['var'] = stasClus[cluster][1]
        adata.uns['InfoClus'][f'cluster_{cluster}']['count'] = stasClus[cluster][2]
        adata.uns['InfoClus'][f'cluster_{cluster}']['ic'] = np.array(icsClus[cluster])
        adata.uns['InfoClus'][f'cluster_{cluster}']['attributes'] = attsClus[cluster]

    adata.var['prior_mean'] = stasPrior[:,0]
    adata.var['prior_var'] = stasPrior[:,1]
    adata.var['feature_type'] = featureType

    adata.write(adata_path)

