import os
import time
import anndata as ad
import warnings
warnings.filterwarnings("ignore")

from os.path import join
from si import ExclusOptimiser
from utils import load_data
from static_visual import painting

DATA_SET_NAME = 'immune'
# DATA_SET_NAME = 'uci_adult'
# DATA_SET_NAME = 'german_socio_eco'

DATA_FOLDER = f'C:/Users/Administrator/OneDrive - UGent/Documents/Data/ExClus/{DATA_SET_NAME}'
WORK_FOLDER = f'../data/{DATA_SET_NAME}'

DATA_FILE = f'{DATA_SET_NAME}.csv'
adata = ad.read_h5ad(f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}/{DATA_SET_NAME}.h5ad')

#  load and rearrange columns of data
path_to_data_file = join(DATA_FOLDER, DATA_FILE)
print("load data ... ", end='')
df_data, df_data_scaled, lenBinary = load_data(path_to_data_file)
print("done")

alpha = 250
beta = 1.5

for EMB_NAME in adata.obsm.keys():
    if EMB_NAME == "tSNE_5":
        tic = time.time()
        embedding = adata.obsm.get(EMB_NAME)
        optimiser = ExclusOptimiser(df_data, df_data_scaled, lenBinary, embedding, alpha=alpha, beta=beta,
                                    name=DATA_SET_NAME, emb_name=EMB_NAME, work_folder=WORK_FOLDER)
        optimiser.optimise(runtime_id=0)
        # print(optimiser._res_in_brief)
        optimiser.save_adata(data_folder=f'C:/Users/Administrator/trace/data/{DATA_SET_NAME}')
        toc = time.time()
        print(f'Time: {toc - tic} s')

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
            painting(DATA_SET_NAME, WORK_FOLDER, file_to_painting, df_data)
            IfVisual = input('\n visualization ExClus result? y/n: ')
















