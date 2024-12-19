import pandas as pd
import os
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE
from tsne import compute_tsne_series
# from umap import UMAP


def get_var_complexity(var: pd.DataFrame) -> pd.Series:
    '''
    Get the complexity of the variable based on the type of the variable
    :param var_type: the type of the variable
    :return: the complexity of the variable
    '''
    # if the variable is numeric, it is considered 2, if it is categorical, it is considered as the count of distinct values
    return var['var_type'].apply(lambda x: 2 if x == 'numeric' else len(var['var_type'].unique()))

def generate_adata(relative_data_path: str, dataset_name: str):
    '''
    Generate an AnnData object and save it as an h5ad file. AnnData object is used widely in this project as a role of saving meta data and computed infoclus result.

    Datails: given dataset name, generating adata file contains
    1. versions of dataset, including raw and scaled
    2. feature types, like numerical or categorical
    3. feature complexities, basically the minimum satitics needed to leaen the feature (mean&var for Guassian)
    3. embeddings of tsne and PCA
    
    Note: to make this function work, you will need the directory structure to be the same as the one in the repo
    '''
    # Load data
    data_path = os.path.join(relative_data_path, dataset_name, f'{dataset_name}.csv')
    df_data = pd.read_csv(data_path)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_data)
    
    # generate adata
    adata = ad.AnnData(X = data_scaled)
    adata.layers['raw_data'] = df_data.values
    adata.layers['scaled_data'] = data_scaled
    adata.var_names = df_data.columns.astype(str)
    # todo: tweak the below to be compatible with all data sets
    adata.var['var_type'] = 'numeric'
    adata.var['var_complexity'] = get_var_complexity(adata.var)
    # todo: check the difference of tsne & PCA between here and Edith used. 
    tsne = TSNE(n_components=2)
    adata.obsm['tsne'] = tsne.fit_transform(data_scaled)
    pca = PCA(n_components=2)
    adata.obsm['pca'] = pca.fit_transform(data_scaled)
    # # TODO: fix umap to work successfully
    # tsne_embs = compute_tsne_series(data = data_scaled,
    #                 fine_exag_iter=[(1, 200)],
    #                 hd_metric= "euclidean",
    #                 init= adata.obsm['pca'],
    #                 sampling_frac=1, # no need to subsample for this small dataset
    #                 smoothing_perplexity=30,
    #                 random_state=42
    # )
    # for exag, emb in tsne_embs.items():
    #     adata.obsm[f"tSNE_{exag}"] = emb

    # save adata
    adata.write(os.path.join(relative_data_path, dataset_name, f'{dataset_name}.h5ad'))
    print(f"adata file is saved at {os.path.join(relative_data_path, dataset_name, f'{dataset_name}.h5ad')}")
    return adata
