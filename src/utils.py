import numpy as np
import copy
import pandas as pd

from sklearn.preprocessing import StandardScaler


def load_data(path):
    # load all data, and reorder the columns of the data according to the attributes
    #     :return data and counts of binary attributes
    data = pd.read_csv(path)
    scaler = StandardScaler()
    data_scaled = copy.deepcopy(data)
    data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled[data_scaled.columns])

    # record information of attributes types and count
    targetsLen = len(data.columns)
    reorder = []
    len_binary = 0
    for i in range(targetsLen):
        col = data.iloc[:, i]
        # Bernoulli for binary
        n_1 = (col == 1).sum()
        if n_1 == np.count_nonzero(col):
            reorder.insert(0, i)
            len_binary += 1
        # Gaussian otherwise
        else:
            reorder.append(i)

    #  do the reorder
    data = data.iloc[:, reorder]
    data_scaled = data_scaled.iloc[:, reorder]

    return data, data_scaled, len_binary
