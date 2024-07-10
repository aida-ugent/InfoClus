import time
import anndata as ad

from os.path import join
from si import ExclusOptimiser
from utils import load_data


'''
immune data
'''
WORK_FOLDER = '../../data/immune'
DATA_SET_NAME = 'immune'
DATA_FILE = 'immune.csv'
adata = ad.read_h5ad(f'{WORK_FOLDER}/{DATA_SET_NAME}.h5ad')


#  load data
path_to_data_file = join(WORK_FOLDER, DATA_FILE)
print("load data ... ", end='')
df_data, df_data_scaled, lenBinary = load_data(path_to_data_file)
print("done")

# for each embedding, store si-clustering and si-explanation
for EMB_NAME in adata.obsm.keys():

    if EMB_NAME == "tSNE_5":
        tic = time.time()
        embedding = adata.obsm.get(EMB_NAME)
        optimiser = ExclusOptimiser(df_data, df_data_scaled, lenBinary, embedding,
                                    name=DATA_SET_NAME, emb_name=EMB_NAME, work_folder = WORK_FOLDER)
        optimiser.optimise(runtime_id=5)
        optimiser.save_adata()
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
                optimiser.save_adata()
                toc = time.time()
                print(f'Time: {toc - tic} s')
            if run_type == 'recalc':
                tic = time.time()
                optimiser.optimise(alpha, beta, ref_runtime_id)
                optimiser.save_adata()
                toc = time.time()
                print(f'Time: {toc - tic} s')
            if_continue = input('\n continue ExClus y/n: ')












