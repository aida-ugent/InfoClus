import copy
import os

import numpy as np
import pandas as pd

from typing import Dict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from utils import get_git_root

def kl_gaussian(m1, s1, m2, s2, epsilon=0.00001):
    # kl(custer||prior)

    mean1 = copy.copy(m1)
    std1 = copy.copy(s1)
    mean2 = copy.copy(m2)
    std2 = copy.copy(s2)

    std1 += epsilon
    std2 += epsilon
    a = np.log(std2 / std1)
    zeros_std2 = std2 == 0
    a[zeros_std2] = 0
    b = (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2)
    return a + b - 1 / 2

def kl_bernoulli(p_value, q_value, epsilon=0.00001):

    p = copy.copy(p_value)
    q = copy.copy(q_value)

    negative_p = p < 0
    negative_q = q < 0
    p[negative_p] = 0
    q[negative_q] = 0
    larger_p = p > 1
    larger_q = q > 1
    p[larger_p] = 1
    q[larger_q] = 1

    zeros_q = q == 0
    q[zeros_q] = epsilon
    ones_q = q == 1
    q[ones_q] = 1 - epsilon

    zeros_p = p == 0
    p[zeros_p] = epsilon
    ones_p = p == 1
    p[ones_p] = 1 - epsilon

    a = p * np.log(p / q)
    b = (1 - p) * np.log((1 - p) / (1 - q))

    zeros_p = p == 0
    a[zeros_p] = 0
    ones_p = p == 1
    b[ones_p] = 0

    return a + b

def get_scaled_data(data: pd.DataFrame, replace_nan: float) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame:
    - Standardizes numeric columns using StandardScaler.
    - Encodes categorical columns (string type) into integers using factorize,
      and then applies StandardScaler to the encoded values.

    Parameters:
        data (pd.DataFrame): Input data containing numeric and categorical columns.

    Returns:
        pd.DataFrame: Transformed data with standardized numeric columns
                      and scaled categorical columns.
    """
    # replace nan to replace_nan
    data = data.fillna(0)
    # Initialize an empty DataFrame to store processed data
    factorized_data = None
    ls_mapping_chain_by_col = None
    scaled_data = pd.DataFrame()

    for col in data.columns:
        col_data = data[col].values
        if col_data.dtype == 'object':  # Check if the column is of string type
            if factorized_data is None and ls_mapping_chain_by_col is None:
                factorized_data = pd.DataFrame()
                ls_mapping_chain_by_col = []
            df_mapping = pd.DataFrame(columns=['raw', 'factorized', 'scaled'])
            unique_values = list(set(col_data.tolist()))
            df_mapping['raw'] = unique_values
            factorized_data[col], uniques = pd.factorize(col_data)
            df_mapping['factorized'] = [np.where(uniques == value)[0][0] for value in df_mapping['raw']]
            scaler = StandardScaler()
            scaled_data[col] = scaler.fit_transform(factorized_data[col].values.reshape(-1, 1)).flatten()
            mapping = {factorized: scaled for factorized, scaled in zip(factorized_data[col], scaled_data[col])}
            df_mapping['scaled'] = df_mapping['factorized'].map(mapping)
            ls_mapping_chain_by_col.append(df_mapping)

        elif pd.api.types.is_numeric_dtype(col_data):  # Check if the column is numeric
            scaler = StandardScaler()
            scaled_data[col] = scaler.fit_transform(data[[col]].values.reshape(-1, 1)).flatten()
        else:
            # Raise an error for unsupported data types
            raise ValueError(f"Unsupported data type in column {col}")
        scaled_data = pd.DataFrame(data=scaled_data,columns = data.columns)
    return factorized_data, ls_mapping_chain_by_col, scaled_data, data

def get_embeddings(data_array: np.ndarray) -> Dict[str, np.ndarray]:
    embeddings_dict = {}
    tsne = TSNE(n_components=2, perplexity=30, random_state=1)
    embeddings_dict['tsne'] = tsne.fit_transform(data_array)
    pca = PCA(n_components=2)
    embeddings_dict['pca'] = pca.fit_transform(data_array)
    return embeddings_dict

def get_var_type_complexity(data: pd.DataFrame, var_type_threshold: int) -> pd.DataFrame:
    data_var_type_complexity = pd.DataFrame(columns=['var_type', 'var_complexity'])
    if 'var_type' in data.columns:
        data_var_type_complexity['var_type'] = data['var_type']
        for col_idx, var_type in enumerate(data['var_type']):
            if var_type == 'numeric':
                data_var_type_complexity.loc[col_idx, 'var_complexity'] = 2
            elif var_type == 'categorical':
                column_data = data.iloc[:, col_idx]
                data_var_type_complexity.loc[col_idx, 'var_complexity'] = column_data.nunique()
            else:
                print('ERROR! Unknown var_type {}'.format(var_type))
                data_var_type_complexity.loc[col_idx, 'var_complexity'] = None
    else:
        for col_idx, col_name in enumerate(data.columns):
            col_data = data.iloc[:, col_idx]
            distinct_counts = col_data.nunique()
            if distinct_counts > var_type_threshold:
                data_var_type_complexity.loc[col_idx, 'var_type'] = 'numeric'
                data_var_type_complexity.loc[col_idx, 'var_complexity'] = 2
            else:
                data_var_type_complexity.loc[col_idx, 'var_type'] = 'categorical'
                data_var_type_complexity.loc[col_idx, 'var_complexity'] = distinct_counts
    return data_var_type_complexity


def get_kde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str, cluster_id: int, cluster_color):
    """
    :return: return kernal desity estimation of one attribute for a cluster
    """
    percentage = len(cluster_att) / len(data_att)

    # Fit KDE models
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': np.linspace(0.1, 5.0, 30)})
    # grid.fit(data_att.reshape(-1, 1))
    # optimal_bandwidth = grid.best_params_['bandwidth']

    kde_data = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(data_att.reshape(-1, 1))
    kde_cluster = KernelDensity(kernel='gaussian', bandwidth=kde_data.bandwidth_).fit(cluster_att.reshape(-1, 1))

    # Generate x values
    x_vals = np.linspace(min(min(data_att), min(cluster_att)), max(max(data_att), max(cluster_att)), 1000)
    kde_data_vals = np.exp(kde_data.score_samples(x_vals.reshape(-1, 1)))
    kde_cluster_vals = np.exp(kde_cluster.score_samples(x_vals.reshape(-1, 1)))

    # Compute overlap density
    cluster_proportion = len(cluster_att) / len(data_att)
    overlap_density = kde_cluster_vals * cluster_proportion

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x_vals, kde_data_vals, label=f'Data', color='black', linewidth=2)
    ax.plot(x_vals, kde_cluster_vals, label=f'C {cluster_id}', color=cluster_color, linestyle='dotted',
            linewidth=4)
    ax.fill_between(x_vals, overlap_density, color=cluster_color, alpha=0.5, label=f'{percentage:.2%}')

    # Labels and legend
    ax.set_xlabel(f"{att_name}", fontsize=25)
    fig.tight_layout()
    # ax.set_ylabel('Density', fontsize=18)
    ax.legend(fontsize=20)
    return fig

def get_barchart(df_mapping_chain, dist_of_fixed_cluster_att, dist_of_att_in_data,  att_id: int, cluster_id: int, att_name: str, cluster_color, overlap: float):

    real_labels = df_mapping_chain.iloc[:, 0]
    dist_pre_cluster_att = pd.Series(dist_of_fixed_cluster_att, index=real_labels)
    dist_prior_per_att = pd.Series(dist_of_att_in_data, index=real_labels)
    sorted_dist_pre_cluster_att = dist_pre_cluster_att.sort_values(ascending=False)
    sorted_dist_prior_per_att = dist_prior_per_att.loc[sorted_dist_pre_cluster_att.index]
    sorted_labels = sorted_dist_pre_cluster_att.index

    x = np.arange(len(sorted_labels))  # Label locations
    width = 0.4  # Width of bars

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - width / 2, sorted_dist_pre_cluster_att, width, label=f'C{cluster_id}-{overlap:.2%}', color=cluster_color)
    ax.bar(x + width / 2, sorted_dist_prior_per_att, width, label=f'Data', color='black')

    ax.set_xlabel(att_name, fontsize=25)
    # ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=20, ha='right')
    ax.legend(fontsize=20)
    # ax.set_title(f"Cluster {cluster_id} - {att_name}")

    plt.tight_layout()
    return fig
#
#
