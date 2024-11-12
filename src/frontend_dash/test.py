import anndata as ad
import pandas as pd
from src.utils import load_data
from src.si import ExclusOptimiser
from dicts2adata import dicts2adata
from layout import config_scatter_graph

dataset_name = 'german_socio_eco'
embedding_name = 'tSNE_1'
alpha=80
beta=1.5
min_att=2
runtime_id=1

adata_path = f'{dataset_name}.h5ad'
adata = ad.read_h5ad(adata_path)

data = adata.layers['raw_data']
df = pd.DataFrame(data, columns=adata.var.index)
df_data, df_scaled, lenBinary = load_data(df)
embedding = adata.obsm.get(embedding_name)

optimiser = ExclusOptimiser(df, df_scaled, lenBinary,
                            embedding, name=dataset_name, emb_name=embedding_name,
                            alpha=alpha, beta=beta, min_att=min_att, max_att=0, runtime_id=runtime_id,
                            work_folder=f'../../data/{dataset_name}')

# todo: finally, integrate the adata write in into exclusoptimiser, or like Zander, first store in optimiser not another file
dicts2adata(dataset_name, adata_path,
            f'../../data/{dataset_name}/{dataset_name}-{embedding_name}-single-{alpha}-{beta}-{min_att}-0-0-0')
updated_adata = ad.read_h5ad(adata_path)

config_scatter_graph(updated_adata.obs['infoclus_clustering'].values,
                                    updated_adata.obsm.get(embedding_name))


print('hello world')