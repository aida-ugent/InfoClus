import copy, json, hashlib
import os

import numpy as np
import pandas as pd

from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from config import Random_State

def kl_gaussian(m1, s1, m2, s2, epsilon=0.00001):
    # kl(custer||prior)

    var1 = copy.copy(s1)
    zeros_var1 = var1 == 0
    var1[zeros_var1] = epsilon
    std1 = var1 ** 0.5

    var2 = copy.copy(s2)
    zeros_var2 = var2 == 0
    var2[zeros_var2] = epsilon
    std2 = var2 ** 0.5

    a = np.log(std2 / std1)
    # zeros_std2 = std2 == 0
    # a[zeros_std2] = 0
    b = (var1 + (m1 - m2) ** 2) / (2 * var2)
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

def get_scaled_data(data: pd.DataFrame, replace_nan: float):
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
    tsne = TSNE(n_components=2, perplexity=30, random_state=Random_State)
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
#
# def get_kde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str, cluster_id: int, cluster_color):
#     """
#     :return: return kernal desity estimation of one attribute for a cluster
#     """
#     percentage = len(cluster_att) / len(data_att)
#
#     # Fit KDE models
#     q_c1 = np.percentile(cluster_att, 25)
#     q_c3 = np.percentile(cluster_att, 75)
#     iqr_c = q_c3 - q_c1
#     q_a1 = np.percentile(data_att, 25)
#     q_a3 = np.percentile(data_att, 75)
#     iqr_a = q_a3 - q_a1
#     min_c = min(np.std(cluster_att), iqr_c / 1.34 + 0.00001)
#     min_a = min(np.std(data_att), iqr_a / 1.34 + 0.00001)
#     bandwidth_c = 0.9 * min_c * cluster_att.shape[0] ** (-0.2)
#     bandwidth_a = 0.9 * min_a * data_att.shape[0] ** (-0.2)
#     bandwidth = max(bandwidth_a, bandwidth_c)
#
#     kde_data = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data_att.reshape(-1, 1))
#     kde_cluster = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(cluster_att.reshape(-1, 1))
#
#     # Generate x values
#     x_vals = np.linspace(min(min(data_att), min(cluster_att)), max(max(data_att), max(cluster_att)), 1000)
#     kde_data_vals = np.exp(kde_data.score_samples(x_vals.reshape(-1, 1)))
#     kde_cluster_vals = np.exp(kde_cluster.score_samples(x_vals.reshape(-1, 1)))
#
#     # Compute overlap density
#     cluster_proportion = len(cluster_att) / len(data_att)
#     overlap_density = kde_cluster_vals * cluster_proportion
#
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.plot(x_vals, kde_cluster_vals, label=f'Cluster {cluster_id}', color=cluster_color, linestyle='dotted',
#             linewidth=4)
#     ax.plot(x_vals, kde_data_vals, label=f'the Whole Data', color='black', linewidth=2)
#     ax.fill_between(x_vals, overlap_density, color=cluster_color, alpha=0.5, label=f'Part of Data covered by Cluster')
#
#     # Labels and legend
#     ax.set_xlabel(f"{att_name}", fontsize=50)
#     ax.set_ylabel('Distribution', fontsize=25)
#     ax.set_yticks([])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     fig.tight_layout()
#     ax.legend(fontsize=16,loc='best')
#     return fig

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
    ax.bar(x - width / 2, sorted_dist_pre_cluster_att, width, label=f'Cluster {cluster_id}', color=cluster_color)
    # ax.bar(x + width / 2, sorted_dist_prior_per_att, width, label=f'Data - -{overlap:.2%} covered by cluster', color='black')
    ax.bar(x + width / 2, sorted_dist_prior_per_att, width, label=f'the Whole Data', color='black')

    ax.set_xlabel(att_name, fontsize=30)
    ax.set_ylabel("Proportion", fontsize=30)
    ax.set_xticks([])
    # ax.set_xticklabels(sorted_labels, rotation=20, ha='right', fontsize=12)
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(fontsize=18, loc='best')
    # ax.set_title(f"Cluster {cluster_id} - {att_name}")

    plt.tight_layout()
    return fig

def get_hashkey_from_dict(dict_obj: dict):
    dict_str = json.dumps(dict_obj, sort_keys=True)
    unique_key = hashlib.md5(dict_str.encode("utf-8")).hexdigest()
    return unique_key

def recur_mean(mean1, count1, mean2, count2):
    # combine two clusters
    # given counts of points in clusters and means of clusters, & return mean of the new cluster within the recursive formula
    return (mean1 * count1 + mean2 * count2) / (count1 + count2)

def recur_var(mean1, var1, count1, mean2, var2, count2):
    # combine two clusters
    # given counts of points in clusters and variances of clusters, also means, & return variance of the new cluster within the recursive formula
    a = (count1 * count2 * (mean2 - mean1) ** 2) / (count1 + count2)
    return (count1 * var1 + count2 * var2 + a) / (count1 + count2)

def recur_meanVar_merge(info_i, info_j):
    count = info_i[2] + info_j[2]
    mean = (info_i[0] * info_i[2] + info_j[0] * info_j[2]) / count
    a = (info_i[2] * info_j[2] * (info_j[0] - info_i[0]) ** 2) / count
    var = (info_i[2] * info_i[1] + info_j[2] * info_j[1] + a) / count
    return [mean, var, count]

def recur_meanVar_remove(mean, var, count, mean1, var1, count1):
    # remove cluster 1 from original cluster, and return mean, variance and count for the left cluster
    count2 = count - count1
    if count2 == 0:
        return None
    mean2 = (count * mean - count1 * mean1) / count2
    var2 = (count * var) / count2 - (count1 * var1) / count2 - (count1 * (mean1 - mean2) ** 2) / count
    negas_var2 = var2 < 0
    var2[negas_var2] = 0
    return [mean2, var2, count2]

def recur_dist_categorical(distribution1: pd.DataFrame, count1: int, distribution2: pd.DataFrame, count2: int) -> pd.DataFrame:
    if (count1 + count2) == 0:
        return None
    distribution3_value = (distribution1.values * count1 + distribution2.values * count2)/(count1 + count2)
    distribution3 = pd.DataFrame(distribution3_value, columns=distribution1.columns)

    return distribution3

def ic_one_info(means_cluster, vars_cluster, n_samples, prior):
    cluster_ic = []
    ic2 = n_samples * kl_gaussian(means_cluster, vars_cluster, prior[0], prior[1])
    cluster_ic.extend(ic2)
    return cluster_ic

def get_opt_attributes(alpha, beta, dls, ics, min_att=2, max_att=5):
    ics = np.array(ics)

    sortedic = np.dstack(np.unravel_index(np.argsort(-ics.ravel()), ics.shape))[0]
    find_index = sortedic[:, 0]
    attributes_total = []
    ic_attributes = 0
    dl = 0
    for i in range(len(ics)):
        index = np.where(find_index == i)[0][0:min_att]
        attributes = [sortedic[ind][1] for ind in index]
        attributes_total.append(attributes)
        ic_attributes += sum(ics[i, attributes])
        dl = dl + sum((dls[attribute]) for attribute in attributes)
        sortedic = np.delete(sortedic, index, axis=0)
        find_index = np.delete(find_index, index, axis=0)
    best_comb_val = ic_attributes / (alpha + dl ** beta)

    out_max_att_limit = False
    while not out_max_att_limit and len(sortedic) > 0:
        extend_cluster_try = sortedic[0][0]
        extend_attr_try = sortedic[0][1]
        sortedic = np.delete(sortedic, 0, axis=0)
        if len(attributes_total[extend_cluster_try]) >= max_att:
            continue
        dl_try = dl + dls[extend_attr_try]
        ic_attributes_try = ic_attributes + ics[extend_cluster_try, extend_attr_try]
        si_try = ic_attributes_try / (alpha + dl_try ** beta)
        if si_try >= best_comb_val:
            best_comb_val = si_try
            attributes_total[extend_cluster_try].append(extend_attr_try)
            dl = dl_try
            ic_attributes = ic_attributes_try
            out_max_att_limit = all(len(attribute) >= max_att for attribute in attributes_total)
        else:
            break

    return attributes_total, ic_attributes, dl, best_comb_val

def create_new_list_by_updating(old_list, new_list_to_change: dict, new_list_to_add: dict):
    list = []
    for i in range(len(old_list)):
        if i in new_list_to_change.keys():
            list.append(new_list_to_change[i])
        else:
            list.append(old_list[i])
    for j in range(len(new_list_to_add)):
        list.append(new_list_to_add[j])
    return list

def get_ic_matrix(clustering: np.ndarray, scaled_data: pd.DataFrame, prior_statistics):

    # initialize information content matrix
    ic_matrix = []

    # get indexes for each cluster label
    index_dict = {}
    for i, val in enumerate(clustering):
        index_dict.setdefault(val, []).append(i)

    # compute statistics and ic for each cluster label
    for cluster_label in sorted(index_dict.keys()):
        cluster = scaled_data.iloc[index_dict[cluster_label]]
        mean_cluster = np.mean(cluster.values, axis=0)
        var_cluster = np.var(cluster.values, axis=0)
        ic_matrix.append(ic_one_info(mean_cluster, var_cluster, len(cluster), prior_statistics))

    return ic_matrix

def calc_optimal_attributes_dl(ic_matrix, dls, alpha, beta, min_att, max_att):
    """
    This is a function used to return optimal attributes for each cluster.
    param
        ics: information content matrix (n*m), where n is count of clusters and m is the number of attributes.
    return
         attributes set for each cluster
    """
    ic_matrix = np.array(ic_matrix)
    attributes_total, ic_attributes, dl, best_comb_val, sortedic = init_optimal_attributes_dl(ic_matrix, dls, alpha, beta, min_att)
    out_max_att_limit = False
    while not out_max_att_limit and len(sortedic) > 0:
        extend_cluster_try = sortedic[0][0]
        extend_attr_try = sortedic[0][1]
        sortedic = np.delete(sortedic, 0, axis=0)
        if len(attributes_total[extend_cluster_try]) >= max_att:
            continue
        dl_try = dl + dls[extend_attr_try]
        ic_attributes_try = ic_attributes + ic_matrix[extend_cluster_try, extend_attr_try]
        si_try = ic_attributes_try / (alpha + dl_try ** beta)
        if si_try >= best_comb_val:
            best_comb_val = si_try
            attributes_total[extend_cluster_try].append(extend_attr_try)
            dl = dl_try
            ic_attributes = ic_attributes_try
            out_max_att_limit = all(len(attribute) >= max_att for attribute in attributes_total)
        else:
            break

    return attributes_total, ic_attributes, dl, best_comb_val

def init_optimal_attributes_dl(ic_matrix, dls, alpha, beta, min_att):

    sortedic = np.dstack(np.unravel_index(np.argsort(-ic_matrix.ravel()), ic_matrix.shape))[0]
    find_index = sortedic[:, 0]
    attributes_total = []
    ic_attributes = 0
    dl = 0
    for i in range(len(ic_matrix)):
        index = np.where(find_index == i)[0][0:min_att]
        attributes = [sortedic[ind][1] for ind in index]
        attributes_total.append(attributes)
        ic_attributes += sum(ic_matrix[i, attributes])
        dl = dl + sum((dls[attribute]) for attribute in attributes)
        sortedic = np.delete(sortedic, index, axis=0)
        find_index = np.delete(find_index, index, axis=0)
    best_comb_val = ic_attributes / (alpha + dl ** beta)

    return attributes_total, ic_attributes, dl, best_comb_val, sortedic

#
# def visualize_result(self, show_now_embedding = True, save_embedding = False, show_now_explanation = False, save_explanation = False):
#
#     # visualize clustering on embedding
#     if self.modify_hierarchical:
#         data = self.datas.data_raw.values
#         labels = self._clustering_opt[self.kmedoids_clustering]
#         embedding = self.all_embeddings[self.emb_name]
#     else:
#         data = self.datas.data.values
#         labels = self._clustering_opt
#         embedding = self.embedding
#     att_names = self.datas.data.columns.values
#     unique_classes = np.unique(labels)
#     num_classes = len(unique_classes)
#
#     colors = sns.color_palette("colorblind", num_classes)  # HUSL generates distinguishable colors
#     fig = plt.figure(figsize=(8, 6))
#     for i, cls in enumerate(unique_classes):
#         # Select points corresponding to the current class
#         class_points = embedding[labels == cls]
#         lable = f'Cluster {cls}'
#         plt.scatter(class_points[:, 0], class_points[:, 1],
#                     color=colors[i], label=lable, s=20)
#     plt.tight_layout()
#
#     num_att = 0
#     for cluster_idx in range(len(self._attributes_opt)):
#         num_att += len(self._attributes_opt[cluster_idx])
#     # plt.text(x=50, y=-50, s=num_att, fontsize=70, fontweight= 'bold', color='black', ha='right', va='bottom')
#     plt.legend(fontsize=16)
#     plt.axis('off')
#     if show_now_embedding:
#         plt.show()
#     if save_embedding:
#         if isinstance(self.model, AgglomerativeClustering):
#             fig_path = f"../figs/embedding_agglomerative_{self.model.linkage}_a{self.alpha}_b{self.beta}-{self.name}_Infoclus"
#             fig_path = fig_path.replace(" ", "_")
#         if isinstance(self.model, KMeans):
#             fig_path = f"../figs/embedding_kmeans_{self.model.n_clusters}_a{self.alpha}_b{self.beta}-{self.name}_Infoclus"
#             fig_path = fig_path.replace(" ", "_")
#         fig.savefig(f'{fig_path}.png')
#
#     # visualize distributions of attributes
#     for cluster_label in unique_classes:
#         instance_cluster_idx = np.where(labels == cluster_label)
#         attributes = self._attributes_opt[cluster_label]
#         cluster = data[instance_cluster_idx]
#         overlap = len(cluster) / len(data)
#         cluster_color  = colors[cluster_label]
#         for att_id in attributes:
#             data_att = data[:, att_id]
#             cluster_att = cluster[:, att_id]
#             att_name = att_names[att_id]
#             att_type = self.var_type[att_id]
#             if att_type == 'categorical':
#                 # todo: clean code here
#                 df_mapping_chain = self.datas.ls_mapping_chain_by_col[att_id]
#                 nuniques = len(df_mapping_chain)
#                 dist_of_fixed_cluster_att = self._clustersRelatedInfo[cluster_label][0].iloc[:nuniques,
#                                             att_id].values
#                 dist_of_att_in_data = self._priors.iloc[:nuniques, att_id].values
#                 fig = utils.get_barchart(df_mapping_chain,dist_of_fixed_cluster_att,dist_of_att_in_data, att_id, cluster_label,att_name, cluster_color, overlap)
#             elif att_type == 'numeric':
#                 fig = utils.get_kde(data_att, cluster_att, att_name, cluster_label, cluster_color)
#             else:
#                 print('unsupported attribute type for visualization:', att_type)
#             if show_now_explanation:
#                 fig.show()
#             if save_explanation:
#                 if isinstance(self.model, AgglomerativeClustering):
#                     fig_path = f"../figs/agglomerative_{self.model.linkage}_a{self.alpha}_b{self.beta}_C{cluster_label}_{overlap:.2}_{att_name}-{self.name}_Infoclus"
#                     fig_path = fig_path.replace(" ", "_")
#                 if isinstance(self.model, KMeans):
#                     fig_path = f"../figs/kmeans_{self.model.n_clusters}_a{self.alpha}_b{self.beta}_C{cluster_label}_{overlap:.2}_{att_name}-{self.name}_Infoclus"
#                     fig_path = fig_path.replace(" ", "_")
#                 fig.savefig(f'{fig_path}.png')
#
