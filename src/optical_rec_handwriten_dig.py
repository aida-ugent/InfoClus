import numpy as np
import anndata as ad
import os
import time
import pandas as pd

from sklearn.manifold import TSNE
from si import ExclusOptimiser
from utils import load_data, load_data_sample
from os.path import join
from static_visual import painting
from ucimlrepo import fetch_ucirepo
from utils import compute_embedding, normalize_embedding, save_emb_adata


# fetch dataset
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
# data (as pandas dataframes)
X = optical_recognition_of_handwritten_digits.data.features
y = optical_recognition_of_handwritten_digits.data.targets

DATA_SET_NAME = 'orhd'
DATA_FOLDER = f'C:/Users/Administrator/OneDrive - UGent/Documents/Data/ExClus/{DATA_SET_NAME}'
WORK_FOLDER = f'../data/{DATA_SET_NAME}'
DATA_FILE = f'{DATA_SET_NAME}.csv'

#  load and rearrange columns of data
print("load data ... ", end='')
df_data, df_data_scaled, lenBinary = load_data(X)
print("done")

EMB_NAME = 'tSNE_1'
adata_file_path = f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}/{DATA_SET_NAME}.h5ad'

embedding = compute_embedding(df_data)
nor_embedding = normalize_embedding(embedding)
save_emb_adata(nor_embedding, EMB_NAME, adata_file_path)

alpha = 400
beta = 1.2
min_att = 2
max_att = 10
runtime_id = 6

if len(df_data.columns) < min_att:
    min_att = len(df_data.columns)

adata = ad.read_h5ad(adata_file_path)
for EMB_NAME in adata.obsm.keys():
    if EMB_NAME == "tSNE_1":

        embedding = adata.obsm.get(EMB_NAME)

        optimiser = ExclusOptimiser(df_data, df_data_scaled,
                                    lenBinary, embedding,name=DATA_SET_NAME, emb_name=EMB_NAME,
                                    alpha=alpha, beta=beta, min_att=min_att, max_att=max_att, runtime_id=runtime_id, work_folder=WORK_FOLDER)
        optimiser.save_adata(data_folder=f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}')
        adata.obs['true_label'] = y.values
        adata.write(adata_file_path)

        if_continue = input('\n continue ExClus y/n: ')
        while if_continue == 'y':
            alpha = float(input('alpha [0, 500]: '))
            beta = float(input('beta [0,2]: '))
            ref_runtime_id = int(input('reference runtime id: {0, 1 ... , 10}: '))
            run_type = input('refine/recalc: ')
            if run_type == 'refine':
                tic = time.time()
                optimiser.refine(alpha, beta, ref_runtime_id)
                optimiser.save_adata(data_folder=f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}')
                toc = time.time()
                print(f'Time: {toc - tic} s')
            if run_type == 'recalc':
                tic = time.time()
                optimiser.optimise(alpha, beta, ref_runtime_id)
                optimiser.save_adata(data_folder=f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}')
                toc = time.time()
                print(f'Time: {toc - tic} s')
            if_continue = input('\n continue ExClus y/n: ')

        IfVisual = input('\n visualization ExClus result? y/n: ')
        while IfVisual == 'y':
            directory = f'../data/{DATA_SET_NAME}'
            all_files = os.listdir(directory)
            print("choose a file to visual from following: ")
            for file_index in range(len(all_files)):
                print(f'{file_index}: {all_files[file_index]}')
            file_index = int(input('\n which file to visualize: '))
            file_to_painting = all_files[file_index]
            output = f"../data/{DATA_SET_NAME}/{file_to_painting}.pdf"
            painting(WORK_FOLDER, file_to_painting, df_data, output)
            IfVisual = input('\n visualization ExClus result? y/n: ')