import pandas as pd

from os.path import join


DATA_SET_NAME = 'cytometry'
n_samples = 2500
state = 2

DATA_FOLDER = f'C:/Users/Administrator/OneDrive - UGent/Documents/Data/ExClus/{DATA_SET_NAME}'
DATA_FILE = f'{DATA_SET_NAME}.csv'
path = join(DATA_FOLDER, DATA_FILE)

data = pd.read_csv(path)
data_sampled = data.sample(n_samples, random_state=state, axis=0)

data_sampled.to_csv(f'{DATA_FOLDER}/cytometry_{n_samples}.csv', index=False)

# import numpy as np
#
# from si import kl_gaussian
#
# a = np.array([1.0, 1.0])
# b = np.array([2.0, 2.0])
# if_continue = input('\n continue y/n: ')
# while if_continue == 'y':
#     print(kl_gaussian(a,a,b,b))
#     if_continue = input('\n continue y/n: ')