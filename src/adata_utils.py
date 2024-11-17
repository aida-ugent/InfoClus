import pandas as pd
import os
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE
from tsne import compute_tsne_series
# from umap import UMAP

def generate_adata(relative_data_path: str, dataset_name: str):
    '''
    Data preprocessing: given dataset name, generating adata file contain versions of dataset, feature types and embeddings
    Note: this function is need the directory structure to be the same as the one in the repo
    :param DATA_SET_NAME:
    :return:
    '''
    # Load data
    data_path = os.path.join(relative_data_path, dataset_name, f'{dataset_name}.csv')
    df_data = pd.read_csv(data_path)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_data)
    
    # generate adata
    adata = ad.AnnData(X = data_scaled)
    adata.layers['raw'] = df_data.values
    adata.layers['scaled'] = data_scaled
    adata.var_names = df_data.columns.astype(str)
    # todo: tweak the below to be compatible with all data sets
    adata.var['var_type'] = 'numeric'
    # todo: check the difference of tsne & PCA between here and Edith used. 
    tsne = TSNE(n_components=2)
    adata.obsm['tsne'] = tsne.fit_transform(data_scaled)
    pca = PCA(n_components=2)
    adata.obsm['pca'] = pca.fit_transform(data_scaled)
    # todo: fix umap to work successfully
    # umap = UMAP(n_components=2)
    # adata.obsm['umap'] = umap.fit_transform(data_scaled)
    tsne_embs = compute_tsne_series(
                    data = data_scaled,
                    fine_exag_iter=[(10, 200), (5, 200), (3, 200), (1, 200)],
                    hd_metric= "euclidean",
                    init= adata.obsm['pca'],
                    sampling_frac=1, # no need to subsample for this small dataset
                    smoothing_perplexity=30,
                    random_state=42
    )
    for exag, emb in tsne_embs.items():
        adata.obsm[f"tSNE_{exag}"] = emb

    # save adata
    adata.write(os.path.join(relative_data_path, dataset_name, f'{dataset_name}.h5ad'))
    print(f"adata file is saved at {os.path.join(relative_data_path, dataset_name, f'{dataset_name}.h5ad')}")
