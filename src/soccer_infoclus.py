import pickle
import copy
import pandas as pd

from sklearn.preprocessing import StandardScaler
from typing import Union
from si import ExclusOptimiser

import warnings
warnings.filterwarnings("ignore")


def load_data_soccer(file: Union[str, pd.DataFrame]):

    if isinstance(file, str):
        data = pd.read_csv(file)
    elif isinstance(file, pd.DataFrame):
        data = file
    scaler = StandardScaler()
    data_scaled = copy.deepcopy(data)
    data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled[data_scaled.columns])

    len_binary = 0

    return data, data_scaled, len_binary


with open("../data/soccer/players_all_info.pkl", "rb") as f:
    players_all_info = pickle.load(f)

player_hd_data = players_all_info.iloc[:, 1:267]
player_embedding = players_all_info[['x', 'y']]


#  load and rearrange columns of data
print("load data ... ", end='')
df_data, df_data_scaled, lenBinary = load_data_soccer(player_hd_data)
print("done")

DATA_SET_NAME = 'soccer'
WORK_FOLDER = f'../data/{DATA_SET_NAME}'

alpha = 300
beta = 1.8
min_att = 2
max_att = 10
runtime_id = 4

EMB_NAME = 'repre_learn'
embedding = player_embedding.values

optimiser = ExclusOptimiser(df_data, df_data_scaled,
                                    lenBinary, embedding,name=DATA_SET_NAME, emb_name=EMB_NAME,
                                    alpha=alpha, beta=beta, min_att=min_att, max_att=max_att, runtime_id=runtime_id, work_folder=WORK_FOLDER)
optimiser.save_adata(data_folder=f'../data/{DATA_SET_NAME}')

print("Test")
