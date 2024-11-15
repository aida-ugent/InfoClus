
import anndata as ad
import os

import pandas as pd

DATA_SET_NAME = 'german_socio_eco'
# get reletive path to access file under data
df = pd.read_csv(f'../data/{DATA_SET_NAME}.csv')

# RELATIVE_DATA_PATH =


#
# def generate_adata(DATA_SET_NAME):
#     '''
#     Data preprocessing: given dataset name, generating adata file contain versions of dataset, feature types and embeddings
#     :param DATA_SET_NAME:
#     :return:
#     '''
#
#
#     adata = ad.AnnData(X=hd_data)
#
#
#
# adata = generate_adata(DATA_SET_NAME)