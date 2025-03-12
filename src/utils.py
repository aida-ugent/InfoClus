import numpy as np
import copy
import pandas as pd
import anndata as ad
import subprocess
import os

from sklearn.preprocessing import StandardScaler
from typing import Union
from sklearn.manifold import TSNE


# Todo: remove the so many versions of load_data, put sample or not as parameter, remove lenBinary
# Todo: generate another readable variable to replace  lenbinary
def load_data(file: Union[str, pd.DataFrame]):
    # load all data, and reorder the columns of the data according to the attributes
    #     :return data and counts of binary attributes
    if isinstance(file, str):
        data = pd.read_csv(file)
    elif isinstance(file, pd.DataFrame):
        data = file
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


def load_data_sample(file: Union[str, pd.DataFrame], n_samples=2500, state=2):
    if isinstance(file, str):
        data = pd.read_csv(file)
    elif isinstance(file, pd.DataFrame):
        data = file
    try:
        data_sampled = data.sample(n_samples, random_state=state, axis=0)
    except:
        data_sampled = data
    scaler = StandardScaler()
    data_scaled = copy.deepcopy(data_sampled)
    data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled[data_scaled.columns])

    # record information of attributes types and count
    targetsLen = len(data_sampled.columns)
    reorder = []
    len_binary = 0
    for i in range(targetsLen):
        col = data_sampled.iloc[:, i]
        # Bernoulli for binary
        n_1 = (col == 1).sum()
        if n_1 == np.count_nonzero(col):
            reorder.insert(0, i)
            len_binary += 1
        # Gaussian otherwise
        else:
            reorder.append(i)

    #  do the reorder
    data_sampled = data_sampled.iloc[:, reorder]
    data_scaled = data_scaled.iloc[:, reorder]


    return data_sampled, data_scaled, len_binary


def load_data_single_type(file: Union[str, pd.DataFrame]):
    # load all data, and reorder the columns of the data according to the attributes
    #     :return data and counts of binary attributes
    if isinstance(file, str):
        data = pd.read_csv(file)
        mappings = {col: pd.factorize(data[col])[1] for col in data.columns}
        data_num = data.apply(lambda col: pd.factorize(col)[0])
    elif isinstance(file, pd.DataFrame):
        data = file
        mappings = {col: pd.factorize(data[col])[1] for col in data.columns}
        data_num = data.apply(lambda col: pd.factorize(col)[0])
    scaler = StandardScaler()
    data_scaled = copy.deepcopy(data_num)
    data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled[data_scaled.columns])

    len_binary = None

    return data, data_scaled, len_binary


def load_data_single_type_sample(file: Union[str, pd.DataFrame], n_samples=2500, state=2):

    if isinstance(file, str):
        data = pd.read_csv(file)
    elif isinstance(file, pd.DataFrame):
        data = file

    try:
        data_sampled = data.sample(n_samples, random_state=state, axis=0)
        data_sampled_num = data_sampled.apply(lambda col: pd.factorize(col)[0])
    except:
        data_sampled = data
        data_sampled_num = data_sampled.apply(lambda col: pd.factorize(col)[0])

    scaler = StandardScaler()
    data_scaled = copy.deepcopy(data_sampled_num)
    data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled[data_scaled.columns])
    len_binary = None

    return data_sampled, data_scaled, len_binary


def compute_embedding(data):

    tsne = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000, learning_rate=200, random_state=2)
    embedding = tsne.fit_transform(data)
    data = {'Y': embedding}

    return data['Y']


def normalize_embedding(arr):
    """
    Normalize the provided embedding to [-1,1].

    Parameters:
    arr (np.ndarray): The embedding to be normalized.

    Returns:
    np.ndarray: The normalized embedding array.
    """
    min = np.min(arr)
    diff = np.max(arr) - min
    arr = (2 * ((arr - min) / diff)) - 1
    return arr


def save_emb_adata(emb: np.ndarray, emb_name: str = '', adata_file_name: str = ''):
    # save a kind of tsne embedding into adata
    adata = ad.read_h5ad(adata_file_name)
    adata.obsm[emb_name] = emb
    adata.uns["methods"]['tSNE'] = np.append(adata.uns["methods"]['tSNE'],emb_name)
    adata.write(adata_file_name)

def get_git_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            universal_newlines=True
        ).strip()
        return root
    except subprocess.CalledProcessError:
        raise RuntimeError("This directory is not a Git repository.")


def get_project_root():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up the directory tree until you find a specific file (like `setup.py` or a config file)
    # In this case, we can stop when we find the root of the project (you can customize the condition)
    while not os.path.exists(os.path.join(current_dir,
                                          'readme.md')):  # You can change 'setup.py' to something else (e.g., README.md or a custom marker file)
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Stop when you reach the root of the filesystem
            raise RuntimeError("Project root not found.")
        current_dir = parent_dir

    return current_dir