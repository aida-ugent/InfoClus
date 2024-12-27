import pickle
import warnings

import pandas as pd
import numpy as np
import os
import collections
import time
import copy
import math

from sklearn.cluster import AgglomerativeClustering
from caching import from_cache, to_cache
import infoclus_utils as utils
from utils import get_git_root

RUNTIME_OPTIONS = [0.01, 0.5, 1, 5, 10, 30, 60, 180, 300, 600, 1800, 3600, np.inf]
VAR_TPYE_THRESHOLD = 20
REPLACE_NAN = 0
EPSILON= 0.00001

ROOT = get_git_root()
DATA_FOLDER = os.path.join(ROOT, 'data')

class InfoClus:
    '''
        InfoClus is a class that offers a explanable clustering solution for datasets
        It uses a pre-clustering model to offer candidate clusters

        usually, the process of InfoClus is as follows:
        1. initialization
        2. optimise: either run InfoClus or read from cache
        3. run InfoClus (optional)

        save object to pickle file

    '''

    ######################################## step 1: initialization ########################################
    def __init__(self, dataset_name: str, data_folder: str = None,
                 main_emb: str = 'tsne',
                 model = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None),
                 alpha: int = None, beta: float = 1.5, min_att: int = 2, max_att: int = 10, runtime_id: int = 3
                 ):
        '''
        The initialization will 
        1. obtain preliminary processed information (like embeddings)
        2. further process information.
            2.1 train model to get agglomerative clustering
            2.2 get linkage matrix and calculate the distribution of each node, 
                parent of each node, points in each node and the values of each attribute
            2.3 prior information computation

        :param name: name of the dataset
        :param data_folder: the path to the data folder
        :param main_emb: the embedding type to be used for clustering
        :param model: the pre-clustering model to be used to offer candidate clusters
        '''
        self.name = dataset_name
        if data_folder is None:
            self.dataset_folder = os.path.join(DATA_FOLDER, dataset_name)
        else:
            self.dataset_folder = data_folder
        self.emb_name = main_emb
        self.model = model
        self.beta = beta
        self.epsilon = EPSILON
        self.min_att = min_att
        self.max_att = max_att
        self.runtime_id = runtime_id
        self.runtime = RUNTIME_OPTIONS[runtime_id]
        self.cache_path = os.path.join(self.dataset_folder, 'cache')

        #################################### step1: obtain preliminary processed information (like embeddings) #########################################
        df_data = pd.read_csv(os.path.join(self.dataset_folder, f'{self.name}.csv'))
        factorized_data, ls_mapping_chain_by_col, self.data_scaled, self.data_raw = utils.get_scaled_data(df_data, REPLACE_NAN)
        if factorized_data is not None and ls_mapping_chain_by_col is not None:
            self.factorized_data = factorized_data
            self.ls_mapping_chain_by_col = ls_mapping_chain_by_col
        self.data = self.data_scaled.values
        if alpha is None:
            self.alpha = int(len(self.data)/10)
        else:
            self.alpha = alpha
        self.all_embeddings = utils.get_embeddings(self.data)
        if self.emb_name not in self.all_embeddings.keys():
            print('Error! embedding not found!')
        else:
            self.embedding = self.all_embeddings[self.emb_name]
        # TODO: change name of _dls to be more cohesive with the code
        df_var_type_complexity = utils.get_var_type_complexity(self.data_raw, VAR_TPYE_THRESHOLD)
        self.var_type = df_var_type_complexity['var_type']
        self._dls = df_var_type_complexity['var_complexity']
        if len(self.var_type.unique()) > 1:
            self.global_var_type = 'mixed'
        else:
            self.global_var_type = self.var_type.iloc[0]

        self._parents = None
        self._linkage_matrix = None
        self._prior_dataset = None
        self._nodesToPoints = {}
        if self.global_var_type == 'mixed':
            print("Error! mixed variables are not supported yet!")
            pass
        elif self.global_var_type == 'numeric':
            self._meansForNodes = {}
            self._varsForNodes = {}
        elif self.global_var_type == 'categorical':
            self._distributionsForNodes = {}
        else:
            print("ERROR! not supported variable type!")

        #################################### step2: furthur process #########################################
        self._fit_model()
        self._create_linkage() 
        self._calc_priors()

        print('initialization done')

    def _fit_model(self):
        '''
        Fit the pre-clustering model
        :return:
        '''
        self.model.fit(self.embedding)

    def _create_linkage(self):
        # TODO: rewrite, redundant code, not clear naming
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        parents = np.full(self.model.children_.shape[0], -1)

        # build empty distribution
        if self.global_var_type == 'categorical':
            count_of_uniques_per_attribute = [len(df) for df in self.ls_mapping_chain_by_col]
            np_data = np.zeros((max(count_of_uniques_per_attribute), len(self.data_raw.columns)))
            mask = np.arange(np_data.shape[0])[:, None] >= np.array(count_of_uniques_per_attribute)
            np_data[mask] = self.epsilon
            empty_distribution = pd.DataFrame(np_data, columns=self.data_raw.columns)

        # for each merge in agglomerative clustering, do computation
        for i, merge in enumerate(self.model.children_):

            leafPoints = []
            self._nodesToPoints[i] = leafPoints

            # left child in merge
            left_child = merge[0]
            current_count_left = 0  # points count of left child node
            # compute count, mean, vars, parent
            if left_child < n_samples:
                current_count_left += 1
                if self.global_var_type == 'mixed':
                    pass
                elif self.global_var_type == 'numeric':
                    m_left = self.data_scaled.iloc[left_child].to_numpy()
                    var_left = np.zeros_like(m_left)
                elif self.global_var_type == 'categorical':
                    left_point = self.data_scaled.iloc[left_child]
                    m_left = empty_distribution.copy()
                    for j_column in range(len(self.data_raw.columns)):
                        att_value = left_point.values[j_column]
                        i_row = self.ls_mapping_chain_by_col[j_column].loc[self.ls_mapping_chain_by_col[j_column]['scaled'] == att_value].index[0]
                        m_left.iloc[i_row, j_column] = 1
                leafPoints.append(left_child)
            else:
                current_count_left += counts[left_child - n_samples]
                parents[left_child - n_samples] = i  # correction by Fuyin Lai
                if self.global_var_type == 'mixed':
                    pass
                elif self.global_var_type == 'numeric':
                    m_left = self._meansForNodes.get(left_child - n_samples)
                    var_left = self._varsForNodes.get(left_child - n_samples)
                elif self.global_var_type == 'categorical':
                    m_left = self._distributionsForNodes.get(left_child - n_samples)
                leafPoints.extend(self._nodesToPoints[left_child - n_samples])
            # right child
            right_child = merge[1]
            current_count_right = 0
            # count, mean, vars, parent
            if right_child < n_samples:
                current_count_right += 1
                if self.global_var_type == 'mixed':
                    pass
                elif self.global_var_type == 'numeric':
                    m_right = self.data_scaled.iloc[right_child].to_numpy()
                    var_right = np.zeros_like(m_right)
                elif self.global_var_type == 'categorical':
                    right_point = self.data_scaled.iloc[right_child]
                    m_right = empty_distribution.copy()
                    for j_column in range(len(self.data_raw.columns)):
                        att_value = right_point.values[j_column]
                        i_row = self.ls_mapping_chain_by_col[j_column].loc[self.ls_mapping_chain_by_col[j_column]['scaled'] == att_value].index[0]
                        m_right.iloc[i_row, j_column] = 1
                leafPoints.append(right_child)
            else:
                current_count_right += counts[right_child - n_samples]
                parents[right_child - n_samples] = i  # correction by Fuyin Lai
                if self.global_var_type == 'mixed':
                    pass
                elif self.global_var_type == 'numeric':
                    m_right = self._meansForNodes.get(right_child - n_samples)
                    var_right = self._varsForNodes.get(right_child - n_samples)
                elif self.global_var_type == 'categorical':
                    m_right = self._distributionsForNodes.get(right_child - n_samples)
                leafPoints.extend(self._nodesToPoints[right_child - n_samples])

            # new mean, var and count for node i
            if self.global_var_type == 'mixed':
                pass
            elif self.global_var_type == 'numeric':
                meanForNode = self.recur_mean(m_left, current_count_left,
                                              m_right, current_count_right)
                self._meansForNodes[i] = meanForNode
                varForNode = self.recur_var(m_left, var_left, current_count_left,
                                            m_right, var_right, current_count_right)
                self._varsForNodes[i] = varForNode
            elif self.global_var_type == 'categorical':
                distForNode = self.recur_dist_categorical(m_left, current_count_left,
                                              m_right, current_count_right)
                self._distributionsForNodes[i] = distForNode
            counts[i] = current_count_left + current_count_right

        # update self
        self._parents = parents  # without counting original points
        self._linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts])

    def _calc_priors(self):
        # TODO: rewrite, remove dl_indices for numeric
        if self.global_var_type == 'mixed':
            pass
        elif self.global_var_type == 'numeric':
            self._priors = np.array([self._meansForNodes[len(self.data) - 2], self._varsForNodes[len(self.data) - 2]]).T
            self._priorsGausM = self._meansForNodes[len(self.data) - 2]
            self._priorsGausS = self._varsForNodes[len(self.data) - 2]
            # Order attribute indices per dl to use later in dl optimisation
            unique_dls = sorted(set(self._dls))
            # Attributes indices split per dl, used to split IC into submatrix and later to find IC value of attribute
            self._dl_indices = collections.OrderedDict()
            for dl in unique_dls:
                # Fill dl_indices for one dl value
                indices = [i for i, value in enumerate(self._dls) if value == dl]
                self._dl_indices[dl] = indices
        elif self.global_var_type == 'categorical':
            self._priors = self._distributionsForNodes[len(self.data) - 2]

    # TODO: remove all recur functions to another file, because they are not related to the class
    def recur_mean(self, mean1, count1, mean2, count2):
        # combine two clusters
        # given counts of points in clusters and means of clusters, & return mean of the new cluster within the recursive formula
        return (mean1 * count1 + mean2 * count2) / (count1 + count2)

    def recur_var(self, mean1, var1, count1, mean2, var2, count2):
        # combine two clusters
        # given counts of points in clusters and standard variances of clusters, also means, & return standard variance of the new cluster within the recursive formula
        a = (count1 * count2 * (mean2 - mean1) ** 2) / (count1 + count2)
        return (count1 * var1 + count2 * var2 + a) / (count1 + count2)

    def recur_meanVar_merge(self, info_i, info_j):
        count = info_i[2] + info_j[2]
        mean = (info_i[0] * info_i[2] + info_j[0] * info_j[2]) / count
        a = (info_i[2] * info_j[2] * (info_j[0] - info_i[0]) ** 2) / count
        var = (info_i[2] * info_i[1] + info_j[2] * info_j[1] + a) / count
        return [mean, var, count]

    def recur_meanVar_remove(self, mean, var, count, mean1, var1, count1):
        # remove cluster 1 from original cluster, and return mean, variance and count for the left cluster
        count2 = count - count1
        if count2 == 0:
            return None
        mean2 = (count * mean - count1 * mean1) / count2
        var2 = (count * var) / count2 - (count1 * var1) / count2 - (count1 * (mean1 - mean2) ** 2) / count
        negas_var2 = var2 < 0
        var2[negas_var2] = 0
        return [mean2, var2, count2]

    def recur_dist_categorical(self, distribution1: pd.DataFrame, count1: int, distribution2: pd.DataFrame, count2: int) -> pd.DataFrame:
        if (count1 + count2) == 0:
            return None
        distribution3_value = (distribution1.values * count1 + distribution2.values * count2)/(count1 + count2)
        distribution3 = pd.DataFrame(distribution3_value, columns=distribution1.columns)

        return distribution3

    # given means, vars, n_samples of a cluster, return its ic, vectorization for attributes
    # todo: not revised yet
    def ic_one_info(self, means_cluster, vars_cluster, n_samples):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warningNum = len(w)
            stds_cluster = vars_cluster ** 0.5
            if len(w) != warningNum:
                breakpoint()
        # stds_cluster = vars_cluster ** 0.5
        cluster_ic = []
        if self.global_var_type == 'mixed':
            pass
        elif self.global_var_type == 'categorical':
            means_binary_cluster = means_cluster
            ic1 = n_samples * utils.kl_bernoulli(means_binary_cluster, self._priorsBernM)
            cluster_ic.extend(ic1)
        elif self.global_var_type == 'numeric':
            means_gaussian_cluster = means_cluster
            stds_gaussian_cluster = stds_cluster
            ic2 = n_samples * utils.kl_gaussian(means_gaussian_cluster, stds_gaussian_cluster,
                                                                   self._priorsGausM,
                                                                   self._priorsGausS)
            cluster_ic.extend(ic2)

        return cluster_ic

    def kl_categorical(self, distribution_cluster: np.ndarray, epsilon: float = 0.00001) -> np.ndarray:
        # kl(p||q) = kl(cluster||prior)
        # kl(p||q) = kl(cluster||prior)

        p = copy.copy(distribution_cluster)
        q = copy.copy(self._priors.values)
        p_safe = np.where(p <= 0, epsilon, p)

        kl_mid_value = p * np.log(p_safe/q)
        kl = np.sum(kl_mid_value, axis=0)

        return kl

    # todo: not revised yet, delete or remove kl_categorical to other places
    def ic_categorical(self, distribution_cluster: pd.DataFrame, size_cluster: int) -> np.ndarray:
        ic = size_cluster * self.kl_categorical(distribution_cluster.values)
        return ic

    ######################################## step 2: optimise: either run InfoClus or read from cache ########################################
    def optimise(self, alpha=None, beta=None, min_att=None, max_att=None, runtime_id=3):
        '''
        optimise result with current hyperparameters, the process is as follows:
        1. update hyperparameters of self
        2. check cache
        3. start clustering when no cache
        4. print the clustering result
        '''
        # update hyperparameters of self
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if min_att is not None:
            self.min_att = min_att
        if max_att is not None:
            self.max_att = max_att
        if runtime_id is not None:
            self.runtime_id = runtime_id
        self.runtime = RUNTIME_OPTIONS[runtime_id]

        # check cache
        cache_name, previously_calculated = self.check_cache()
        # start clustering when no cache
        if previously_calculated is None:
            self._si_opt = 0
            self._res_in_brief = ''
            self._run_infoclus()
            self.create_cache_version(cache_name)
        file_path = os.path.join(self.dataset_folder, f'{self.name}_{self.emb_name}.pkl')
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

        print(f'\nInfoClus - Dataset: {self.name} Emb: {self.emb_name} Alpha: {self.alpha} Beta: {self.beta} Ref. Runtime: {self.runtime}')
        print(f'Count of Clusters: {len(set(self._clustering_opt))}')
        clusters = list(range(self._clusterlabel_max + 1))
        for cluster_idx in range(len(self._attributes_opt)):
            print(f"    cluster {cluster_idx}:")
            print(f'        count of points: {sum(self._clustering_opt == cluster_idx)}')
            print(f'        attributes: ', end='')
            for j in self._attributes_opt[cluster_idx]:
                print(f'{self.data_raw.columns[j]} ', end='')
            print("")
        print("SI: ", self._si_opt)


        return self._clustering_opt, self.embedding

    ######################################## step 3: run InfoClus ########################################
    def _run_infoclus(self):
        '''
        Here is the core part of Infoclus algorithm, the process is as follows:
        1. initialization of all result-related variables as None
        2. iteration preparation and start iteration of splitting in limited time

        Note: one split means enumerating all possible splits(nodes) and choose the best one to split one cluster into two
        '''
        #################################### step1: initialization result-related variables #########################################
        self._clustering_opt = None  # final clustering labels for each point
        self._si_opt = 0  # value of si for this clustering
        self._clustersRelatedInfo = {}  # means, vars, and counts for each cluster
        self._clusterlabel_max: int = 0  # maximum label, from 0
        self._attributes_opt = None  # chosen attributes for each cluster
        self._ic_opt = None  # ic of all attributes for each cluster
        self._total_ic_opt = 0
        self._total_dl_opt = 0  # value for summing up length of attributes

        self._split_nodes_opt = []  # splitted nodes and their classification label, tuple inside
        self._nodes_opt = None  # the left nodes that could be used for further splitting

        self._split_nodes_opt.append(("others", 0))
        clustering_new_info = {}

        if self.global_var_type == 'mixed':
            pass
        elif self.global_var_type == 'numeric':
            clustering_new_info[0] = [self._priors[:, 0], self._priors[:, 1], len(self.data)]
        elif self.global_var_type == 'categorical':
            clustering_new_info[0] = [self._priors, len(self.data_scaled)]

        #################################### step2: iteration #########################################
        # nodes: possible splits (generating by combining nodes and their parents)
        nodes_idx = range(len(self._linkage_matrix) - 1)  # count from 0, without leaf points
        parents = self._parents[:-1]  # count from 0, without leaf points
        nodes = [[x, y] for x, y in zip(nodes_idx, parents)]
        clustering_new = None
        ic_new = None
        local_optimum = False
        considered_splitting_times = 0
        start = time.time()
        print("splitting start ... ", end='')
        while nodes and (time.time() - start < self.runtime):
            considered_splitting_times += 1
            # get the best node to split
            nodes, clustering_new, attributes_new, si_val_new, ic_new, ic_att_new, dl_new, opt_node, clustering_new_info = self._choose_optimal_split(
                nodes,
                clustering=clustering_new,
                max_cluster_label=considered_splitting_times - 1,
                clusteringInfo=clustering_new_info,
                ic_temp=ic_new)
            # if the best node in this iteration is better than current record
            if si_val_new > self._si_opt:
                if local_optimum:
                    print("Local opt")
                    print("Clusters: ", len(set(self._clustering_opt)))
                    print("SI: ", self._si_opt)
                #TODO: figure out the mechanism of copy and deepcopy in python, so that the time could be saved
                self._clustering_opt = copy.deepcopy(clustering_new)
                self._clusterlabel_max = max(set(self._clustering_opt))
                # self._clusterlabel_max += 1
                # if self._clusterlabel_max != max(set(self._clustering_opt)):
                #     raise Exception('self._clusterlabel_max != len(set(self._clustering_opt))')
                new_label = self._clusterlabel_max
                self._split_nodes_opt.append((opt_node[0], new_label))
                self._clustersRelatedInfo = copy.deepcopy(clustering_new_info)
                self._attributes_opt = copy.deepcopy(attributes_new)
                self._si_opt = si_val_new
                self._ic_opt = copy.deepcopy(ic_new)
                self._total_dl_opt = dl_new
                self._total_ic_opt = ic_att_new
                self._nodes_opt = copy.deepcopy(nodes)
            else:
                local_optimum = True
        print("done")
        print(f'considered splitting times: {considered_splitting_times}')

    def _choose_optimal_split(self, nodes, clustering=None, clusteringInfo=None, max_cluster_label=0, ic_temp=None):
        '''
        enumerate all possible splits(nodes) and choose the best one. 
        
        For each node, the process is as follows:
        1. seperate current clustering based on node
        2. compute ics of all features for each cluster
        3. get the best attributes set for each cluster based on ics, also si if we split based on this node
        4. update the best node in this for loop

        After considering all nodes and get the best one, we update nodes(for further splitting)
        '''
        largest_si = 0
        largest_attributes = None
        largest_idx = None
        largest_clustering = None
        largest_ic = None
        largest_dl = 0
        largest_ic_attributes = 0
        largest_before_split = None
        largest_parent = None
        largest_clusteringInfo = None

        # find the node that takes the highest SI
        for node_idx, parent in nodes:
            #################################### step1: split the clustering based on node #########################################
            new_clustering, new_cluster, old_cluster, idx_new, idx_old = self._node_indices_split(node_idx,
                                                                                                  pre_index=clustering,
                                                                                                  max_label=max_cluster_label)
            
            #################################### step2: compute ics of all features for each cluster #########################################
            # get infor (mean, var, count) of new clustering
            before_split = np.append(idx_old, idx_new)
            new_clusteringInfo = copy.deepcopy(clusteringInfo)
            clusterInfo = new_clusteringInfo.get(old_cluster)
            if self.global_var_type == 'mixed':
                pass
            elif self.global_var_type == 'numeric':
                nodeInfo = [self._meansForNodes.get(node_idx), self._varsForNodes.get(node_idx), len(idx_new)]
                otherInfo = self.recur_meanVar_remove(clusterInfo[0], clusterInfo[1], clusterInfo[2],
                                                      nodeInfo[0], nodeInfo[1], nodeInfo[2])
                if otherInfo == None:
                    continue
            elif self.global_var_type == 'categorical':
                nodeInfo = [self._distributionsForNodes.get(node_idx), len(idx_new)]
                otherInfo = [self.recur_dist_categorical(clusterInfo[0], clusterInfo[1],
                                                      nodeInfo[0], -nodeInfo[1]), clusterInfo[1] - nodeInfo[1]]
                if otherInfo[1] == 0:
                    continue
            new_clusteringInfo[old_cluster] = otherInfo
            new_clusteringInfo[new_cluster] = nodeInfo
            # get ic for new clustering based on infor
            if clustering is None:
                ics = []
                if self.global_var_type == 'mixed':
                    pass
                elif self.global_var_type == 'numeric':
                    ics.append(self.ic_one_info(otherInfo[0], otherInfo[1], otherInfo[2]))
                    ics.append(self.ic_one_info(nodeInfo[0], nodeInfo[1], nodeInfo[2]))
                elif self.global_var_type == 'categorical':
                    ics.append(self.ic_categorical(otherInfo[0], otherInfo[1]))
                    ics.append(self.ic_categorical(nodeInfo[0], nodeInfo[1]))
            else:
                ics = copy.copy(ic_temp)
                if self.global_var_type == 'mixed':
                    pass
                elif self.global_var_type == 'numeric':
                    ics[old_cluster] = self.ic_one_info(otherInfo[0], otherInfo[1], otherInfo[2])
                    ics.append(self.ic_one_info(nodeInfo[0], nodeInfo[1], nodeInfo[2]))
                elif self.global_var_type == 'categorical':
                    ics[old_cluster] = self.ic_categorical(otherInfo[0], otherInfo[1])
                    ics.append(self.ic_categorical(nodeInfo[0], nodeInfo[1]))

            #################################### step3: get the best attributes set for each cluster based on ics #########################################
            attributes, ic_attributes, dl, si_val = self.calc_optimal_attributes_dl(ics)

            #################################### step4: update the best node in this for loop #########################################
            if si_val > largest_si:
                largest_si = si_val
                largest_attributes = attributes
                largest_idx = node_idx
                largest_parent = parent
                largest_clustering = new_clustering
                largest_clusteringInfo = new_clusteringInfo
                largest_ic = ics
                largest_dl = dl
                largest_ic_attributes = ic_attributes
                largest_before_split = before_split

        #################################### update nodes(for further splitting) #########################################
        # remove nodes based on the best node chosen
        nodes.remove([largest_idx, largest_parent])
        delete_node = largest_parent
        delete_parent = self._parents[delete_node]
        # remove parent node, so when we split, we do not actually merge
        while delete_parent != -1 and nodes.__contains__([delete_node, delete_parent]):
            nodes.remove([delete_node, delete_parent])
            delete_node = delete_parent
            delete_parent = self._parents[delete_node]

        return nodes, largest_clustering, largest_attributes, largest_si, largest_ic, largest_ic_attributes, largest_dl, [
            largest_idx, largest_before_split, largest_parent, 0], largest_clusteringInfo
    
    def _node_indices_split(self, node_idx, pre_index=None, max_label=0):
        '''
        change indices after splitting one node out into a new cluster
        '''
        # get pre index of clustering
        n_samples = len(self.model.labels_)
        indices = copy.deepcopy(pre_index)
        if pre_index is None:
            indices = np.zeros(n_samples, dtype=int)
        # get index that are going to change
        new_cluster = max_label + 1
        to_change = self._nodesToPoints[node_idx]
        old_cluster = indices[to_change[0]]
        # change indices
        indices[to_change] = new_cluster
        not_change = np.where(indices == old_cluster)

        return indices, new_cluster, old_cluster, to_change, not_change

    # get the best attribute for each cluster
    def _init_optimal_attributes_dl(self, ics):
    # todo: not revised yet
        sortedic = np.dstack(np.unravel_index(np.argsort(-ics.ravel()), ics.shape))[0]
        find_index = sortedic[:, 0]
        attributes_total = []
        ic_attributes = 0
        dl = 0
        for i in range(len(ics)):
            index = np.where(find_index == i)[0][0:self.min_att]
            attributes = [sortedic[ind][1] for ind in index]
            attributes_total.append(attributes)
            ic_attributes += sum(ics[i, attributes])
            dl = dl + sum((self._dls.iloc[attribute]) for attribute in attributes)
            sortedic = np.delete(sortedic, index, axis=0)
            find_index = np.delete(find_index, index, axis=0)
        best_comb_val = ic_attributes / (self.alpha + dl ** self.beta)

        return attributes_total, ic_attributes, dl, best_comb_val, sortedic

    def calc_optimal_attributes_dl(self, ics):
        '''
        return attributes set for each cluster
        '''
        ics = np.array(ics)
        attributes_total, ic_attributes, dl, best_comb_val, sortedic = self._init_optimal_attributes_dl(ics)
        out_max_att_limit = False
        while not out_max_att_limit and len(sortedic) > 0:
            extend_cluster_try = sortedic[0][0]
            extend_attr_try = sortedic[0][1]
            sortedic = np.delete(sortedic, 0, axis=0)
            if len(attributes_total[extend_cluster_try]) >= self.max_att:
                continue
            dl_try = dl + self._dls.iloc[extend_attr_try]
            ic_attributes_try = ic_attributes + ics[extend_cluster_try, extend_attr_try]
            si_try = ic_attributes_try / (self.alpha + (dl_try) ** self.beta)
            if si_try >= best_comb_val:
                best_comb_val = si_try
                attributes_total[extend_cluster_try].append(extend_attr_try)
                dl = dl_try
                ic_attributes = ic_attributes_try
                out_max_att_limit = all(len(attribute) >= self.max_att for attribute in attributes_total)
            else:
                break

        return attributes_total, ic_attributes, dl, best_comb_val
    
    def create_cache_version(self, cache_name):
        previously_calculated = {"clustering": self._clustering_opt,
                                 "split": self._split_nodes_opt,
                                 "maxlabel": self._clusterlabel_max,
                                 "global_var_type": self.global_var_type,
                                 "infor": self._clustersRelatedInfo,
                                 "attributes": self._attributes_opt,
                                 "prior": self._priors,
                                 "si": self._si_opt,
                                 "ic": self._ic_opt,
                                 "dls": self._dls,
                                 "nodes": self._nodes_opt,
                                 "total_dl": self._total_dl_opt,
                                 "total_ic": self._total_ic_opt,
                                 "res_in_brief": self._res_in_brief
                                 }
        to_cache(os.path.join(self.cache_path, cache_name), previously_calculated)

    def check_cache(self):
        cache_name = f'{self.name}_{self.emb_name}_{self.model.linkage}_alpha{int(self.alpha)}_beta{self.beta}_mina{self.min_att}_maxa{self.max_att}_runid{int(self.runtime_id)}'
        previously_calculated = from_cache(os.path.join(self.cache_path, cache_name))
        if previously_calculated is not None:
            print("From cache")
            self._clustering_opt = previously_calculated["clustering"]
            self._split_nodes_opt = previously_calculated["split"]
            self._clusterlabel_max = previously_calculated["maxlabel"]
            self.global_var_type = previously_calculated["global_var_type"]
            self._clustersRelatedInfo = previously_calculated["infor"]
            self._attributes_opt = previously_calculated["attributes"]
            self._priors = previously_calculated["prior"]
            self._si_opt = previously_calculated["si"]
            self._ic_opt = previously_calculated["ic"]
            self._dls = previously_calculated["dls"]
            self._nodes_opt = previously_calculated["nodes"]
            self._total_dl_opt = previously_calculated["total_dl"]
            self._total_ic_opt = previously_calculated["total_ic"]
            self._res_in_brief = previously_calculated["res_in_brief"]
        return cache_name, previously_calculated

    # # TODO: rename the variables in adata to match frontend
    # def update_adata(self):
    #
    #     import anndata as ad
    #     file_name = os.path.join(self.dataset_folder, f'{self.name}.h5ad')
    #     adata = ad.read_h5ad(file_name)
    # inf
    #     adata.obs['infoclus_clustering'] = self._clustering_opt
    #     adata.uns['InfoClus'] = {}
    #     adata.uns['InfoClus']['si'] = self._si_opt
    #     adata.uns['InfoClus']['main_emb'] = self.emb_name
    #     adata.uns['InfoClus']['hyperparameters'] = {}
    #     adata.uns['InfoClus']['hyperparameters']['alpha'] = self.alpha
    #     adata.uns['InfoClus']['hyperparameters']['beta'] = self.beta
    #     adata.uns['InfoClus']['hyperparameters']['mina'] = self.min_att
    #     adata.uns['InfoClus']['hyperparameters']['maxa'] = self.max_att
    #     adata.uns['InfoClus']['hyperparameters']['runid'] = self.runtime_id
    #     for cluster in range(self._clusterlabel_max+1):
    #         # todo: decide whether to store statitics based on raw data instead of scaled data
    #         adata.uns['InfoClus'][f'cluster_{cluster}'] = {}
    #         adata.uns['InfoClus'][f'cluster_{cluster}']['mean'] = self._clustersRelatedInfo[cluster][0]
    #         adata.uns['InfoClus'][f'cluster_{cluster}']['var'] = self._clustersRelatedInfo[cluster][1]
    #         adata.uns['InfoClus'][f'cluster_{cluster}']['count'] = self._clustersRelatedInfo[cluster][2]
    #         adata.uns['InfoClus'][f'cluster_{cluster}']['ic'] = np.array(self._ic_opt[cluster])
    #         adata.uns['InfoClus'][f'cluster_{cluster}']['attributes'] = self._attributes_opt[cluster]
    #     adata.var['prior_mean'] = self._priors[:,0]
    #     adata.var['prior_var'] = self._priors[:,1]
    #     adata.write(file_name)
    #     print(f'update {file_name} successfully')