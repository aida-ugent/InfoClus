import math

import numpy as np
import copy
import collections
import queue
import time
import warnings
import cProfile
import pstats

from collections import deque
from hashlib import sha256
from queue import Queue
from os.path import join

import pandas as pd

from caching import from_cache, to_cache
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import to_tree


RUNTIME_OPTIONS = [0.01, 0.5, 1, 5, 10, 30, 60, 180, 300, 600, 1800, 3600, np.inf]
IfProfile = False

IfLimitAtt = True
linkages = ['ward', 'complete', 'average', 'single']
linkage = linkages[3]
sampleWeight = 1

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




class ExclusOptimiser:

    #  Initialize this class and do some basic computation
    def __init__(self,
                 df, df_scaled,
                 lenBinary, embedding, name=None, emb_name="tSNE",
                 model=AgglomerativeClustering(linkage=linkage, distance_threshold=0, n_clusters=None),
                 alpha=250, beta=1.6, min_att=3, max_att=10, runtime_id=0, work_folder="."
                 ):
        self.name = name
        self.emb_name = emb_name
        self.data = df
        self.data_scaled = df_scaled
        self.embedding = embedding
        self._binaryTargetsLen = lenBinary
        self._samplesWeight = sampleWeight
        if self._binaryTargetsLen == None:
            self._allAttType = 'categorical'
            if self._allAttType == 'categorical':
                self._valuesOfAttributes = []
                self._maxValuesOfAttributes = 0
                self._fixedDl = 0
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.00001
        self.min_att = min_att
        self.max_att = max_att
        self.runtime = RUNTIME_OPTIONS[runtime_id]
        self._targetsLen = len(self.data.columns)
        self.cache_path = work_folder

        hash_string, previously_calculated = self.check_cache()
        if previously_calculated != None:
            print(
                f'-------------------- ExClus - Dataset: {self.name} Emb: {self.emb_name} Alpha: {self.alpha} Beta: {self.beta} Ref. Runtime: {self.runtime}\n')
            print(f'Clusters: {len(set(self._clustering_opt))}\n')
            clusters = list(range(self._clusterlabel_max + 1))
            column_names = self.data.columns
            for i in clusters:
                print(f"  cluster {i}: {sum(self._clustering_opt == i)} points")
                print(f'    attributes: ', end='')
                for j in self._attributes_opt[i]:
                    print(f'{column_names[j]} ', end='')
                print("")
            print("SI: ", self._si_opt)
        else:
            # clusterNode related
            self._clusterTree_root = None
            self._nodesToPoints = {}  # non-leaf nodes To points under these nodes, ranging from 0
            if self._binaryTargetsLen != None:
                self._meansForNodes = {}
                self._varsForNodes = {}
            else:
                self._distributionsForNodes = {}

            # calculation
            self._fit_model()  # get agglomarative clustering and get linage matrix
            self._calc_priors()

            # storing the optimal clustering and related information
            self._clustering_opt = None  # indices
            self._split_nodes_opt = []  # splitted nodes and their classification label, tuple inside
            self._clusterlabel_max: int = 0  # maximum label, from 0
            self._clustersRelatedInfo = {}  # means, vars, and counts for each cluster
            self._attributes_opt = None  # chosen attributes for each cluster
            self._ic_opt = None  # ic of all attributes for each cluster
            self._nodes_opt = None  # the left nodes that could be used for further splitting
            self._si_opt = 0  # value of si for this clustering
            self._total_dl_opt = 0  # value for summing up length of attributes
            self._total_ic_opt = 0  # value for summing up all ic used
            self._res_in_brief = '' # string variable to record clustering result in brief

            # time related variables
            self.TIME1_chooseOptimalSplit = 0
            self.time_nodesEnumeration = 0
            self.time_removingNodes = 0
            self.time_infor_1 = 0
            self.time_infor_2_icOneInfo = 0
            self.time_infor_2_append = 0
            self.time_infor_2_deepcopy = 0
            self.time_infor_3 = 0
            self.TIME1_1_calcOptimalAttributesDl = 0
            self.TIME1_4_icOneInfo = 0
            self.count1_4 = 0
            self.TIME1_1_1_initOptimalAttributesDl = 0

            self.optimise(runtime_id=runtime_id)

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

    # update self 1. _parents, 2. _linkage_matrix and 3. _clusterTree_root
    def _create_linkage(self):

        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        parents = np.full(self.model.children_.shape[0], -1)

        # initialize possible values of all attributes
        for column in self.data_scaled.columns:

            if self._binaryTargetsLen == None and self._allAttType == 'categorical':

                possible_values = list(self.data_scaled[column].factorize()[1])
                self._valuesOfAttributes.append({value: index for index, value in enumerate(possible_values)})

                if len(possible_values) > self._maxValuesOfAttributes:
                    self._maxValuesOfAttributes = len(possible_values)

        # build empty distribution
        if self._binaryTargetsLen == None:
            if self._allAttType == 'categorical':
                np_data = np.zeros((self._maxValuesOfAttributes, self.data.columns.size))
                a = np.array([len(df) for df in self._valuesOfAttributes])
                mask = np.arange(np_data.shape[0])[:, None] >= a
                np_data[mask] = self.epsilon
                empty_distribution = pd.DataFrame(np_data, columns=self.data.columns)

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
                if self._binaryTargetsLen != None:
                    m_left = self.data_scaled.iloc[left_child].to_numpy()
                    var_left = np.zeros_like(m_left)
                elif self._allAttType == 'categorical':
                    left_point = self.data_scaled.iloc[left_child]
                    m_left = empty_distribution.copy()
                    for j_column in range(self._targetsLen):
                        att_value = left_point.values[j_column]
                        i_row = self._valuesOfAttributes[j_column].get(att_value)
                        m_left.iloc[i_row, j_column] = 1
                leafPoints.append(left_child)
            else:
                current_count_left += counts[left_child - n_samples]
                parents[left_child - n_samples] = i  # correction by Fuyin Lai
                if self._binaryTargetsLen != None:
                    m_left = self._meansForNodes.get(left_child - n_samples)
                    var_left = self._varsForNodes.get(left_child - n_samples)
                elif self._allAttType == 'categorical':
                    m_left = self._distributionsForNodes.get(left_child - n_samples)
                leafPoints.extend(self._nodesToPoints[left_child - n_samples])
            # right child
            right_child = merge[1]
            current_count_right = 0
            # count, mean, vars, parent
            if right_child < n_samples:
                current_count_right += 1
                if self._binaryTargetsLen != None:
                    m_right = self.data_scaled.iloc[right_child].to_numpy()
                    var_right = np.zeros_like(m_right)
                elif self._allAttType == 'categorical':
                    right_point = self.data_scaled.iloc[right_child]
                    m_right = empty_distribution.copy()
                    for j_column in range(self._targetsLen):
                        att_value = right_point.values[j_column]
                        i_row = self._valuesOfAttributes[j_column].get(att_value)
                        m_right.iloc[i_row, j_column] = 1
                leafPoints.append(right_child)
            else:
                current_count_right += counts[right_child - n_samples]
                parents[right_child - n_samples] = i  # correction by Fuyin Lai
                if self._binaryTargetsLen != None:
                    m_right = self._meansForNodes.get(right_child - n_samples)
                    var_right = self._varsForNodes.get(right_child - n_samples)
                elif self._allAttType == 'categorical':
                    m_right = self._distributionsForNodes.get(right_child - n_samples)
                leafPoints.extend(self._nodesToPoints[right_child - n_samples])

            # new mean, var and count for node i
            if self._binaryTargetsLen != None:
                meanForNode = self.recur_mean(m_left, current_count_left,
                                              m_right, current_count_right)
                self._meansForNodes[i] = meanForNode
                varForNode = self.recur_var(m_left, var_left, current_count_left,
                                            m_right, var_right, current_count_right)
                self._varsForNodes[i] = varForNode
            elif self._allAttType == 'categorical':
                distForNode = self.recur_dist_categorical(m_left, current_count_left,
                                              m_right, current_count_right)
                self._distributionsForNodes[i] = distForNode
            counts[i] = current_count_left + current_count_right

        # update self
        self._parents = parents  # without counting original points
        self._linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts])
        self._clusterTree_root = to_tree(self._linkage_matrix, rd=False)

    def _fit_model(self):
        self.model.fit_predict(self.embedding)
        self._create_linkage()

    # update self 1. _priors 2._priorsBernM 3._priorsGausM 4._priorsGausS
    def _calc_priors(self):

        if self._binaryTargetsLen != None:
            self._priors = np.array([self._meansForNodes[len(self.data) - 2], self._varsForNodes[len(self.data) - 2]]).T
            self._priorsBernM = self._meansForNodes[len(self.data) - 2][0: self._binaryTargetsLen]
            self._priorsGausM = self._meansForNodes[len(self.data) - 2][self._binaryTargetsLen:]
            self._priorsGausS = self._varsForNodes[len(self.data) - 2][self._binaryTargetsLen:]

            a = np.ones(self._binaryTargetsLen, dtype=int)
            b = np.ones(self._targetsLen - self._binaryTargetsLen, dtype=int) + np.ones(
                        self._targetsLen - self._binaryTargetsLen, dtype=int)
            self._dls = np.append(a, b)

            # Order attribute indices per dl to use later in dl optimisation
            unique_dls = sorted(set(self._dls))
            # Attributes indices split per dl, used to split IC into submatrix and later to find IC value of attribute
            self._dl_indices = collections.OrderedDict()
            for dl in unique_dls:
                # Fill dl_indices for one dl value
                indices = [i for i, value in enumerate(self._dls) if value == dl]
                self._dl_indices[dl] = indices

        elif self._allAttType == 'categorical':
            self._priors = self._distributionsForNodes[len(self.data) - 2]
            self._dls = [len(attribute) for attribute in self._valuesOfAttributes]

    # given means, vars, n_samples of a cluster, return its ic, vectorization for attributes
    def ic_one_info(self, means_cluster, vars_cluster, n_samples):

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warningNum = len(w)
            stds_cluster = vars_cluster ** 0.5
            if len(w) != warningNum:
                breakpoint()

        # stds_cluster = vars_cluster ** 0.5
        cluster_ic = []

        # ic for all Bernoulli targets
        if self._binaryTargetsLen != 0:
            means_binary_cluster = means_cluster[0: self._binaryTargetsLen]
            ic1 = (n_samples ** self._samplesWeight) * kl_bernoulli(means_binary_cluster, self._priorsBernM)
            cluster_ic.extend(ic1)

        # ic for all Gaussian targets
        if self._binaryTargetsLen != self._targetsLen:
            means_gaussian_cluster = means_cluster[self._binaryTargetsLen:]
            stds_gaussian_cluster = stds_cluster[self._binaryTargetsLen:]
            ic2 = (n_samples ** self._samplesWeight) * kl_gaussian(means_gaussian_cluster, stds_gaussian_cluster, self._priorsGausM,
                                          self._priorsGausS)
            cluster_ic.extend(ic2)

        return cluster_ic

    def kl_categorical(self, distribution_cluster: np.ndarray, epsilon: float = 0.00001) -> np.ndarray:
        # kl(p||q) = kl(cluster||prior)

        p = copy.copy(distribution_cluster)
        q = copy.copy(self._priors.values)
        p_safe = np.where(p <= 0, epsilon, p)

        kl_mid_value = p * np.log(p_safe/q)
        kl = np.sum(kl_mid_value, axis=0)

        return kl

    def ic_categorical(self, distribution_cluster: pd.DataFrame, size_cluster: int) -> np.ndarray:
        ic = (size_cluster ** self._samplesWeight) * self.kl_categorical(distribution_cluster.values)
        return ic

    # change indices after splitting one node out into a new cluster
    def _node_indices_split(self, node_idx, pre_index=None, max_label=0):
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
        tic = time.time()

        if IfLimitAtt:
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
                if self._binaryTargetsLen != None:
                    dl = dl + sum(1 if attribute<self._binaryTargetsLen else 2 for attribute in attributes)
                elif self._allAttType == 'categorical':
                    dl = dl + sum((self._fixedDl + self._dls[attribute]) for attribute in attributes)
                sortedic = np.delete(sortedic, index, axis=0)
                find_index = np.delete(find_index, index, axis=0)
            best_comb_val = ic_attributes / (self.alpha + dl ** self.beta)

            if self._binaryTargetsLen != None:
                ics_dl = collections.OrderedDict()
                ics_dl[1], ics_dl[2] = [], []
                for sort in sortedic:
                    cluster = sort[0]
                    att = sort[1]
                    if att < self._binaryTargetsLen:
                        ics_dl[1].append(sort)
                    else:
                        new_sort = [cluster, att-self._binaryTargetsLen]
                        ics_dl[2].append(new_sort)
            elif self._allAttType == 'categorical':
                ics_dl = sortedic
        else:
            # Which type of dl exist
            unique_dls = sorted(set(self._dls))
            min_dl = unique_dls[0]

            # Each element is matrix where per row first element is cluster, second element is index to use in dl_indices
            # Order by increasing IC in a queue
            # Used to decide which attribute should be tried next
            # Will first be used to ensure each cluster has at least one attribute
            ics_dl = collections.OrderedDict()

            # Which attributes should be considered to be the first in each cluster
            # Ensure each cluster has at least one
            to_check_first = np.zeros((len(ics), len(unique_dls), 3), dtype=np.int16)

            # Fill dl_indices and ics_dl
            double_test = []
            for val in unique_dls:
                # Attribute indices for dl (val)
                indices = self._dl_indices[val]
                # Only ics for indices are used
                icssub = ics[:, indices]
                # Sort the submatrix according to ic index, this wil be put into ics_dl later as a queue
                sortedic = np.dstack(np.unravel_index(np.argsort(-icssub.ravel()), icssub.shape))[0]
                ics_dl[val] = sortedic

                # Find all attributes that should be considered to check as a first attribute for each cluster
                find_index = sortedic[:, 0]
                for i in range(len(ics)):
                    index = np.where(find_index == i)[0][0]
                    attribute = indices[sortedic[index][1]]
                    if val == unique_dls[0]:
                        to_check_first[i][val - min_dl] = np.array([attribute, val, index])
                    elif ics[i][to_check_first[i][val - min_dl - 1][0]] < ics[i][attribute]:
                        double_test.append(
                            [val, i, ics[i][to_check_first[i][val - min_dl - 1][0]], to_check_first[i][val - min_dl - 1][1],
                             ics[i][attribute]])
                        to_check_first[i][val - min_dl] = np.array([attribute, val, index])

            double_test.sort(key=lambda x: x[4], reverse=True)
            best_combination = to_check_first[:, 0]
            # Total IC only including attributes used for explanation
            ic_attributes = sum(ics[np.arange(len(ics)), best_combination[:, 0]])
            # Total DL for clustering
            dl = sum(best_combination[:, 1])
            best_comb_val = ic_attributes / (self.alpha + dl ** self.beta)

            iterate = True
            while iterate:
                iterate = False
                delete = []
                for i, row in enumerate(double_test):
                    dl_attribute = row[0]
                    attribute = to_check_first[row[1], dl_attribute - min_dl]
                    ic_test = ic_attributes + row[4] - row[2]
                    dl_test = dl + dl_attribute - row[3]
                    val_test = ic_test / (self.alpha + dl_test ** self.beta)
                    if val_test > best_comb_val:
                        ic_attributes = ic_test
                        dl = dl_test
                        best_comb_val = val_test
                        best_combination[row[1]] = attribute
                        delete.append(i)
                        iterate = True
                for i in sorted(delete, reverse=True):
                    del double_test[i]

            # Remove all attributes that have already been added
            # Put remaining ones in queues for ea
            for key, sortedic in ics_dl.items():
                to_delete = best_combination[best_combination[:, 1] == key]
                to_add = np.delete(sortedic, to_delete[:, 2], 0)
                list_object = to_add.tolist()
                ics_dl[key] = list_object

            # Add attributes such that each cluster has one attribute at least
            # Attributes used to explain each cluster (row = cluster)
            attributes_total = [[value[0]] for value in best_combination]

        toc = time.time()
        self.TIME1_1_1_initOptimalAttributesDl += toc - tic
        return attributes_total, ic_attributes, dl, best_comb_val, ics_dl

    # return attributes set for each cluster
    def calc_optimal_attributes_dl(self, ics):

        tic = time.time()
        ics = np.array(ics)

        # get one attribute for each cluster
        attributes_total, ic_attributes, dl, best_comb_val, ics_dl = self._init_optimal_attributes_dl(ics)

        # Optimise
        if self._binaryTargetsLen != None:
            si = -1
            new_value = best_comb_val
            ic_temp = 0
            dl_temp = 0
            current_ic_index = 0
            while new_value > si:
                # New becomes old
                si = new_value
                # Check passed so update attributes, ic, and total dl + remove chosen attribute from its queue
                if si != best_comb_val:
                    attr = ics_dl[dl_temp][current_ic_index]
                    current_ic_index += 1
                    attributes_total[attr[0]].append(self._dl_indices[dl_temp][attr[1]])
                    dl += dl_temp
                    ic_attributes += ic_temp
                # Look for next attribute to test
                ic_temp = 0
                new_temp = 0
                dl_temp = 0
                # Check in order of increasing dl which attribute to add
                for key, value in ics_dl.items():
                    try:
                        test_att = value[current_ic_index]
                    except:
                        continue
                    ic_test = ics[test_att[0]][self._dl_indices[key][test_att[1]]]
                    # Only check att with higher dl if ic higher
                    if ic_test < ic_temp:
                        continue
                    si_test = (ic_attributes + ic_test) / (self.alpha + (dl + key) ** self.beta)
                    if si_test > new_temp:
                        new_temp = si_test
                        ic_temp = ic_test
                        dl_temp = key
                new_value = new_temp

        elif self._allAttType == 'categorical':

            si = best_comb_val
            left_chances = 2

            for row in ics_dl:

                test_cluster = row[0]
                test_att = row[1]
                ic_test = ics[test_cluster][test_att]
                si_test = (ic_attributes + ic_test) / (
                            self.alpha + (dl + self._fixedDl + self._dls[test_att]) ** self.beta)

                if si_test > si:
                    attributes_total[test_cluster].append(test_att)
                    ic_attributes = ic_attributes + ic_test
                    dl = dl + self._fixedDl + self._dls[test_att]
                    si = si_test
                else:
                    left_chances = left_chances - 1

                if left_chances < 0:
                    break

        toc = time.time()
        self.TIME1_1_calcOptimalAttributesDl += toc - tic
        return attributes_total, ic_attributes, dl, si

    # choose the best node to split
    def _choose_optimal_split(self, nodes, clustering=None, clusteringInfo=None, max_cluster_label=0, ic_temp=None):

        tic = time.time()

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
        tic1 = time.time()
        for node_idx, parent in nodes:
            # split out node_idx based on pre_index
            new_clustering, new_cluster, old_cluster, idx_new, idx_old = self._node_indices_split(node_idx,
                                                                                                  pre_index=clustering,
                                                                                                  max_label=max_cluster_label)
            # get infor (mean, var, count) of new clustering
            tic_infor_1 = time.time()
            before_split = np.append(idx_old, idx_new)
            new_clusteringInfo = copy.deepcopy(clusteringInfo)
            clusterInfo = new_clusteringInfo.get(old_cluster)
            if self._binaryTargetsLen != None:
                nodeInfo = [self._meansForNodes.get(node_idx), self._varsForNodes.get(node_idx), len(idx_new)]
                otherInfo = self.recur_meanVar_remove(clusterInfo[0], clusterInfo[1], clusterInfo[2],
                                                      nodeInfo[0], nodeInfo[1], nodeInfo[2])
                if otherInfo == None:
                    continue
            elif self._allAttType == 'categorical':
                nodeInfo = [self._distributionsForNodes.get(node_idx), len(idx_new)]
                otherInfo = [self.recur_dist_categorical(clusterInfo[0], clusterInfo[1],
                                                      nodeInfo[0], -nodeInfo[1]), clusterInfo[1] - nodeInfo[1]]
                if otherInfo[1] == 0:
                    continue
            new_clusteringInfo[old_cluster] = otherInfo
            new_clusteringInfo[new_cluster] = nodeInfo
            toc_infor_1 = time.time()
            self.time_infor_1 += toc_infor_1 - tic_infor_1
            # get ic for new clustering based on infor
            tic_infor_2 = time.time()
            if clustering is None:
                ics = []

                if self._binaryTargetsLen != None:

                    if self._samplesWeight == None:

                        kl_otherCluster = kl_gaussian(otherInfo[0][self._binaryTargetsLen:], otherInfo[1][self._binaryTargetsLen:], self._priorsGausM, self._priorsGausS)
                        kl_nodeCluster = kl_gaussian(nodeInfo[0][self._binaryTargetsLen:], nodeInfo[1][self._binaryTargetsLen:], self._priorsGausM, self._priorsGausS)

                        if np.max(kl_nodeCluster) > np.max(kl_otherCluster):

                            scale_factor = np.max(kl_nodeCluster) / np.max(kl_otherCluster)
                            self._samplesWeight = round(math.log(scale_factor)/math.log(otherInfo[2]/nodeInfo[2]), 1)

                            ics.append((otherInfo[2] ** self._samplesWeight) * kl_otherCluster)
                            ics.append((nodeInfo[2] ** self._samplesWeight) * kl_nodeCluster)

                        elif np.max(kl_nodeCluster) < np.max(kl_otherCluster):

                            scale_factor = np.max(kl_otherCluster) / np.max(kl_nodeCluster)
                            self._samplesWeight = round(math.log(scale_factor) / math.log(nodeInfo[2] / otherInfo[2]), 1)

                            ics.append((otherInfo[2] ** self._samplesWeight) * kl_otherCluster)
                            ics.append((nodeInfo[2] ** self._samplesWeight) * kl_nodeCluster)

                        else:
                            self._samplesWeight = 1

                            ics.append((otherInfo[2] ** self._samplesWeight) * kl_otherCluster)
                            ics.append((nodeInfo[2] ** self._samplesWeight) * kl_nodeCluster)

                    else:

                        ics.append(self.ic_one_info(otherInfo[0], otherInfo[1], otherInfo[2]))
                        ics.append(self.ic_one_info(nodeInfo[0], nodeInfo[1], nodeInfo[2]))

                elif self._allAttType == 'categorical':

                    if self._samplesWeight == None:

                        kl_otherCluster = self.kl_categorical(otherInfo[0].values)
                        kl_nodeCluster = self.kl_categorical(nodeInfo[0].values)

                        if np.max(kl_nodeCluster) > np.max(kl_otherCluster):

                            scale_factor = np.max(kl_nodeCluster) / np.max(kl_otherCluster)
                            self._samplesWeight = round(math.log(scale_factor)/math.log(otherInfo[1]/nodeInfo[1]), 1)

                            ics.append((otherInfo[1] ** self._samplesWeight) * kl_otherCluster)
                            ics.append((nodeInfo[1] ** self._samplesWeight) * kl_nodeCluster)

                        elif np.max(kl_nodeCluster) < np.max(kl_otherCluster):

                            scale_factor = np.max(kl_otherCluster) / np.max(kl_nodeCluster)
                            self._samplesWeight = round(math.log(scale_factor) / math.log(nodeInfo[1] / otherInfo[1]), 1)

                            ics.append((otherInfo[1] ** self._samplesWeight) * kl_otherCluster)
                            ics.append((nodeInfo[1] ** self._samplesWeight) * kl_nodeCluster)

                        else:
                            self._samplesWeight = 1

                            ics.append((otherInfo[1] ** self._samplesWeight) * kl_otherCluster)
                            ics.append((nodeInfo[1] ** self._samplesWeight) * kl_nodeCluster)

                    else:

                        ics.append(self.ic_categorical(otherInfo[0], otherInfo[1]))
                        ics.append(self.ic_categorical(nodeInfo[0], nodeInfo[1]))

            else:

                ics = copy.copy(ic_temp)

                if self._binaryTargetsLen != None:

                    if self._samplesWeight == None:
                        print("Error, when clustering is not none, self._samplesWeight should already be assigned")

                    else:

                        ics[old_cluster] = self.ic_one_info(otherInfo[0], otherInfo[1], otherInfo[2])
                        ics.append(self.ic_one_info(nodeInfo[0], nodeInfo[1], nodeInfo[2]))

                elif self._allAttType == 'categorical':

                    if self._samplesWeight == None:
                        print("Error, when clustering is not none, self._samplesWeight should already be assigned")

                    else:

                        ics[old_cluster] = self.ic_categorical(otherInfo[0], otherInfo[1])
                        ics.append(self.ic_categorical(nodeInfo[0], nodeInfo[1]))

            # get attributes for each cluster
            attributes, ic_attributes, dl, si_val = self.calc_optimal_attributes_dl(ics)

            # update the best node in this for loop
            tic_infor_3 = time.time()
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
            toc_infor_3 = time.time()
            self.time_infor_3 += toc_infor_3 - tic_infor_3
        toc1 = time.time()
        self.time_nodesEnumeration += toc1 - tic1

        # remove nodes based on the best node chosen
        tic2 = time.time()
        nodes.remove([largest_idx, largest_parent])
        delete_node = largest_parent
        delete_parent = self._parents[delete_node]
        # remove parent node, so when we split, we do not actually merge
        while delete_parent != -1 and nodes.__contains__([delete_node, delete_parent]):
            nodes.remove([delete_node, delete_parent])
            delete_node = delete_parent
            delete_parent = self._parents[delete_node]
        toc2 = time.time()
        self.time_removingNodes += toc2 - tic2

        toc = time.time()
        self.TIME1_chooseOptimalSplit += toc - tic

        return nodes, largest_clustering, largest_attributes, largest_si, largest_ic, largest_ic_attributes, largest_dl, [
            largest_idx, largest_before_split, largest_parent, 0], largest_clusteringInfo

    def _iterate_levels(self):

        self._samplesWeight = sampleWeight
        self._clustering_opt = None  # indices
        self._split_nodes_opt = []  # splitted nodes and their classification label, tuple inside
        self._clusterlabel_max: int = 0  # maximum label, from 0
        self._clustersRelatedInfo = {}  # means, vars, and counts for each cluster
        self._attributes_opt = None  # chosen attributes for each cluster
        self._ic_opt = None  # ic of all attributes for each cluster
        self._nodes_opt = None  # the left nodes that could be used for further splitting
        self._si_opt = 0  # value of si for this clustering
        self._total_dl_opt = 0  # value for summing up length of attributes
        self._total_ic_opt = 0

        # nodes: possible splits (generating by combining nodes and their parents)
        nodes_idx = range(len(self._linkage_matrix) - 1)  # count from 0, without leaf points
        parents = self._parents[:-1]  # count from 0, without leaf points
        nodes = [[x, y] for x, y in zip(nodes_idx, parents)]

        clustering_new = None
        self._split_nodes_opt.append(("others", 0))
        clustering_new_info = {}
        if self._binaryTargetsLen != None:
            clustering_new_info[0] = [self._priors[:, 0], self._priors[:, 1], len(self.data)]
        elif self._allAttType == 'categorical':
            clustering_new_info[0] = [self._priors, len(self.data_scaled)]
        ic_new = None
        local_optimum = False
        iterations = 0

        # start iteration of splitting
        start = time.time()
        print("splitting start ... ", end='')
        while nodes and (time.time() - start < self.runtime):
            iterations += 1
            if IfProfile:
                print(f'profile choose_optimal_split in _iterate_levels')
                pr = cProfile.Profile()
                pr.enable()
            # get the best node to split
            nodes, clustering_new, attributes_new, si_val_new, ic_new, ic_att_new, dl_new, opt_node, clustering_new_info = self._choose_optimal_split(
                nodes,
                clustering=clustering_new,
                max_cluster_label=iterations - 1,
                clusteringInfo=clustering_new_info,
                ic_temp=ic_new)
            if IfProfile:
                pr.disable()
                stats = pstats.Stats(pr)
                stats.strip_dirs().sort_stats("cumtime").print_stats(10)
            # if the best node in this iteration is better than current record
            if si_val_new > self._si_opt:
                if local_optimum:
                    print("Local opt")
                    print("Clusters: ", len(set(self._clustering_opt)))
                    print("SI: ", self._si_opt)
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
        iterations_refine = 0
        # print("refine start ... ", end='')
        # iterations_refine = self._iterate_refine()
        # print("done")
        print(f'Iterations {iterations} + {iterations_refine}')
        end_time = time.time()
        self._res_in_brief = f'''
        -------------------- ExClus - Dataset: {self.name} Emb: {self.emb_name} Linkage: {linkage} min_Attr: {self.min_att} Alpha: {int(self.alpha)} Beta: {self.beta} Ref. Runtime: {self.runtime}
        Total Clusters: {int(self._clusterlabel_max+1)}     
        '''
        for cluster in range(self._clusterlabel_max+1):
            str = f'''
            cluster {cluster}: {sum(self._clustering_opt == cluster)} points
            '''
            self._res_in_brief = self._res_in_brief + "\n" + str
        self._res_in_brief += f'''
        SI:  {self._si_opt}
        Time: {end_time - start}
        '''

    # check cache, and get results from cache if it exists
    def check_cache(self, alpha_pre_refine=0, beta_pre_refine=0):
        to_hash = f'{self.name}-{self.emb_name}-{linkage}-{int(self.alpha)}-{self.beta}-{self.min_att}-{int(self.runtime)}-{int(alpha_pre_refine)}-{beta_pre_refine}'
        # hash_string = sha256(to_hash.encode('utf-8')).hexdigest()
        hash_string = to_hash
        previously_calculated = from_cache(join(self.cache_path, hash_string))
        if previously_calculated is not None:
            print("From cache")
            self._clustering_opt = previously_calculated["clustering"]
            self._split_nodes_opt = previously_calculated["split"]
            self._clusterlabel_max = previously_calculated["maxlabel"]
            self._binaryTargetsLen = previously_calculated["binary_att_length"]
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
        return hash_string, previously_calculated

    def create_cache_version(self, hash_string):
        previously_calculated = {"clustering": self._clustering_opt,
                                 "split": self._split_nodes_opt,
                                 "maxlabel": self._clusterlabel_max,
                                 "binary_att_length": self._binaryTargetsLen,
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
        to_cache(join(self.cache_path, hash_string), previously_calculated)

    # start clustering
    def optimise(self, alpha=None, beta=None, runtime_id=2):

        # update hyperparameters of self
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.runtime = RUNTIME_OPTIONS[runtime_id]

        # check cache
        hash_string, previously_calculated = self.check_cache()

        # start clustering when no cache
        if previously_calculated is None:
            self._si_opt = 0
            self._res_in_brief = ''
            self._iterate_levels()
            self.create_cache_version(hash_string)

        print(f'-------------------- ExClus - Dataset: {self.name} Emb: {self.emb_name} Alpha: {self.alpha} Beta: {self.beta} Ref. Runtime: {self.runtime}\n')
        print(f'Clusters: {len(set(self._clustering_opt))}\n')
        clusters = list(range(self._clusterlabel_max + 1))
        column_names = self.data.columns
        for i in clusters:
            print(f"  cluster {i}: {sum(self._clustering_opt == i)} points")
            print(f'    attributes: ', end='')
            for j in self._attributes_opt[i]:
                print(f'{column_names[j]} ', end='')
            print("")
        print("SI: ", self._si_opt)

        return self._clustering_opt, self._attributes_opt, self._si_opt

    def shift_key(self, dict, old_key, new_key):
        res = copy.deepcopy(dict)
        for i, k in enumerate(dict):
            if k in old_key:
                index = old_key.index(k)
                v = dict[k]
                res[new_key[index]] = v
        return res

    def find_closest_ancestor(self, node_i, node_j):
        p_i = node_i
        ps_i = []
        p_j = node_j
        ps_j = []

        # if self._parents[p_i] != -1 and self._parents[p_j] != -1:
        #     if p_i == p_j:
        #         return p_i
        # else:
        #     return None
        while p_i != -1 or p_j != -1:
            if self._parents[p_i] != -1:
                p_i = self._parents[p_i]
                if p_i in ps_j:
                    return p_i
                ps_i.append(p_i)
            try:
                if self._parents[p_j] != -1:
                    p_j = self._parents[p_j]
                    if p_j in ps_i:
                        return p_j
                    ps_j.append(p_j)
            except IndexError:
                print("IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")
        return None

    def _merge(self, related_info, ic_temp, context):

        node_i, node_j, label_i, label_j, ace_node, ace_points, other_points = context[0][0], context[0][1], context[0][
            2], context[0][3], context[0][4], context[0][5], context[0][6]

        # changing: related info, ic
        info_others_old = related_info[0]
        info_others_change = [self.data_scaled.iloc[other_points].mean(), self.data_scaled.iloc[other_points].var(),
                                  len(other_points)]
        info_others_new = self.recur_meanVar_remove(info_others_old[0], info_others_old[1], info_others_old[2],
                                                    info_others_change[0], info_others_change[1], info_others_change[2])
        if info_others_new == None and len(self._split_nodes_opt) == 3:
            return None

        if info_others_new == None:
            context[1] = True

            info_i = related_info[label_i]
            info_j = related_info[label_j]
            info_new_cluster = self.recur_meanVar_merge(self.recur_meanVar_merge(info_i, info_j), info_others_change)
            related_info[min(label_i, label_j)] = info_new_cluster
            ic_temp[min(label_i, label_j)] = self.ic_one_info(info_new_cluster[0], info_new_cluster[1],
                                                              info_new_cluster[2])

            ic_temp.pop(max(label_i, label_j))
            related_info.pop(max(label_i, label_j))

            related_info.pop(0)
            ic_temp.pop(0)

            old_label = list(range(self._clusterlabel_max + 1)[max(label_i, label_j) + 1:])
            new_label = list(range(self._clusterlabel_max + 1)[max(label_i, label_j):])
            related_info = self.shift_key(related_info, old_label, new_label)

            old_label = list(range(1, self._clusterlabel_max))
            new_label = list(range(self._clusterlabel_max - 1))
            related_info = self.shift_key(related_info, old_label, new_label)

        else:
            related_info[0] = info_others_new
            ic_temp[0] = self.ic_one_info(info_others_new[0], info_others_new[1], info_others_new[2])

            info_i = related_info[label_i]
            info_j = related_info[label_j]
            info_new_cluster = self.recur_meanVar_merge(self.recur_meanVar_merge(info_i, info_j), info_others_change)
            related_info[min(label_i, label_j)] = info_new_cluster
            ic_temp[min(label_i, label_j)] = self.ic_one_info(info_new_cluster[0], info_new_cluster[1],
                                                              info_new_cluster[2])

            ic_temp.pop(max(label_i, label_j))
            related_info.pop(max(label_i, label_j))

            old_label = list(range(self._clusterlabel_max + 1)[max(label_i, label_j) + 1:])
            new_label = list(range(self._clusterlabel_max + 1)[max(label_i, label_j):])
            related_info = self.shift_key(related_info, old_label, new_label)

        # get attributes for each cluster
        attributes, ic_attributes, dl, si_val = self.calc_optimal_attributes_dl(ic_temp)

        return (attributes, ic_attributes, dl, si_val)

    def _choose_optimal_merge(self, related_info, ic):

        opt_info = None
        opt_ic = None
        opt_context = None
        opt_attributes = None
        opt_ic_att = 0
        opt_dl = 0
        opt_si = 0

        # get nodes that could be merged together
        merge_nodes = []
        splitted_nodes = self._split_nodes_opt
        len_nodes = len(self._split_nodes_opt)
        for i in range(len_nodes)[1:]:
            for j in range(len_nodes)[i + 1:]:
                node_i, label_i = splitted_nodes[i][0], splitted_nodes[i][1]
                node_j, label_j = splitted_nodes[j][0], splitted_nodes[j][1]
                ace_node = self.find_closest_ancestor(node_i, node_j)
                if ace_node == None:
                    continue
                ace_points = self._nodesToPoints[ace_node]
                try:
                    ace_points_indices = [self._clustering_opt[i] for i in ace_points]
                except:
                    print("error, ace_points_indices = self._clustering_opt[ace_points], TypeError: list indices must be integers or slices, not list")
                ace_points_labels = set(ace_points_indices)
                if len(ace_points_labels) == 3 and label_i in ace_points_labels and label_j in ace_points_labels:
                    if 0 in ace_points_labels:
                        other_points = [x for x in ace_points if
                                        x not in self._nodesToPoints[node_i] and x not in self._nodesToPoints[node_j]]
                        merge_nodes.append(
                            [(node_i, node_j, label_i, label_j, ace_node, ace_points, other_points), False])

        # for each pair of clusters that could be merged together
        for context in merge_nodes:
            related_info_copy = copy.deepcopy(related_info)
            ic_copy = copy.deepcopy(ic)
            res = self._merge(related_info_copy, ic_copy, context)
            if res == None:
                continue
            attributes, ic_attributes, dl, si_val = res[0], res[1], res[2], res[3]
            if si_val > opt_si:
                opt_info = related_info_copy
                opt_ic = ic_copy
                opt_context = context
                opt_attributes = attributes
                opt_ic_att = ic_attributes
                opt_dl = dl
                opt_si = si_val

        return opt_info, opt_ic, opt_context, opt_attributes, opt_ic_att, opt_dl, opt_si

    def _iterate_refine(self):

        s_nodes = copy.deepcopy(self._nodes_opt)
        s_clustering = copy.deepcopy(self._clustering_opt)
        s_max_label = copy.deepcopy(self._clusterlabel_max)
        s_ic = copy.deepcopy(self._ic_opt)
        s_info = copy.deepcopy(self._clustersRelatedInfo)

        m_ic = copy.deepcopy(self._ic_opt)
        m_info = copy.deepcopy(self._clustersRelatedInfo)

        si_opt = self._total_ic_opt / (self.alpha + self._total_dl_opt ** self.beta)

        disable_merge = False
        disable_split = False
        iteration_refine: int = 0
        start = time.time()
        while time.time() - start < self.runtime:
            iteration_refine += 1
            if not disable_merge:
                m_info, m_ic, m_context, m_attributes, m_ic_att, m_dl, m_si = self._choose_optimal_merge(m_info,
                                                                                                         m_ic)
            if not disable_split:
                if IfProfile:
                    print(f'profile choose_optimal_split in _iterate_refine')
                    pr = cProfile.Profile()
                    pr.enable()
                s_nodes, s_clustering, s_attributes, s_si, s_ic, s_ic_att, s_dl, s_opt_node, s_info = self._choose_optimal_split(
                    s_nodes, clustering=s_clustering, max_cluster_label=s_max_label, clusteringInfo=s_info,
                    ic_temp=s_ic)
                if IfProfile:
                    pr.disable()
                    stats = pstats.Stats(pr)
                    stats.strip_dirs().sort_stats("cumtime").print_stats(10)

            if not disable_split and s_si > si_opt and s_si > m_si:

                s_max_label += 1
                # update information
                self._clustering_opt = s_clustering
                self._clusterlabel_max += 1
                if self._clusterlabel_max != max(set(self._clustering_opt)):
                    raise Exception('self._clusterlabel_max != len(set(self._clustering_opt))')
                new_label = self._clusterlabel_max
                self._split_nodes_opt.append((s_opt_node[0], new_label))
                self._clustersRelatedInfo = s_info
                self._attributes_opt = s_attributes
                si_opt = s_si
                self._ic_opt = s_ic
                self._total_dl_opt = s_dl
                self._total_ic_opt = s_ic_att
                self._nodes_opt = s_nodes

                disable_merge = True

            elif not disable_merge and m_si > si_opt:

                # compute updated information
                m_clustering, m_split_nodes, m_cluster_label_max, m_nodes = self.merge_update(m_context)

                # update information
                self._clustering_opt = m_clustering
                self._split_nodes_opt = m_split_nodes
                self._clusterlabel_max = m_cluster_label_max
                if self._clusterlabel_max != max(set(self._clustering_opt)):
                    raise Exception('self._clusterlabel_max != len(set(self._clustering_opt))')
                self._clustersRelatedInfo = m_info
                self._attributes_opt = m_attributes
                self._ic_opt = m_ic
                self._nodes_opt = m_nodes
                si_opt = m_si
                self._total_dl_opt = m_dl
                self._total_ic_opt = m_ic_att

                disable_split = True

            else:
                break

            self._si_opt = si_opt
        end_time = time.time()
        self._clustering_opt = np.array(self._clustering_opt)
        self._res_in_brief = f'''
        -------------------- ExClus - Dataset: {self.name} Emb: {self.emb_name}  Linkage: {linkage} min_Attr: {self.min_att} Alpha: {int(self.alpha)} Beta: {self.beta} Ref. Runtime: {self.runtime}
        Total Clusters: {int(self._clusterlabel_max+1)}     
        '''
        for cluster in range(self._clusterlabel_max+1):
            str = f'''
            cluster {cluster}: {sum(self._clustering_opt == cluster)}
            '''
            self._res_in_brief = self._res_in_brief + "\n" + str
        self._res_in_brief += f'''
        SI:  {self._si_opt}
        Time: {end_time - start}
        '''
        return iteration_refine

    def merge_update(self, context):

        m_clustering = copy.deepcopy(self._clustering_opt)
        split_nodes = copy.deepcopy(self._split_nodes_opt)
        m_cluster_label_max = copy.deepcopy(self._clusterlabel_max)
        m_nodes = copy.deepcopy(self._nodes_opt)

        node_i, node_j, label_i, label_j, ace_node, ace_points, other_points = context[0][0], context[0][1], context[0][
            2], context[0][3], context[0][4], context[0][5], context[0][6]

        if context[1]:
            old_label = list(range(m_cluster_label_max + 1))[max(label_i, label_j) + 1:]
            # m_clustering = list(map(lambda x: x.replace(max(label_i, label_j), min(label_i, label_j)), m_clustering))
            m_clustering[m_clustering == max(label_i, label_j)] = min(label_i, label_j)
            for i in other_points: m_clustering[i] = min(label_i, label_j)
            for i in old_label: m_clustering = [i - 1 if x == i else x for x in m_clustering]
            m_clustering = [x - 1 for x in m_clustering]

            m_split_nodes_temp = [(ace_node, min(label_i, label_j))]
            for nodes in enumerate(split_nodes):
                node, label = nodes[1][0], nodes[1][1]
                if label == label_i or label == label_j:
                    continue
                elif label > max(label_i, label_j):
                    m_split_nodes_temp.append((node, label - 1))
                else:
                    m_split_nodes_temp.append(nodes)
            m_split_nodes = [("others", 0)]
            for nodes in enumerate(m_split_nodes_temp):
                node, label = nodes[1][0], nodes[1][1]
                if label == 1:
                    continue
                else:
                    m_split_nodes.append((node, label - 1))

            m_cluster_label_max = m_cluster_label_max - 2
        else:
            old_label = list(range(m_cluster_label_max + 1))[max(label_i, label_j) + 1:]
            # m_clustering = list(map(lambda x: x.replace(max(label_i, label_j), min(label_i, label_j)), m_clustering))
            m_clustering[m_clustering == max(label_i, label_j)] = min(label_i, label_j)
            for i in other_points: m_clustering[i] = min(label_i, label_j)
            for i in old_label: m_clustering = [i - 1 if x == i else x for x in m_clustering]

            m_split_nodes_temp = [(ace_node, min(label_i, label_j))]
            for nodes in enumerate(split_nodes):
                node, label = nodes[1][0], nodes[1][1]
                if label == label_i or label == label_j:
                    continue
                elif label > max(label_i, label_j):
                    m_split_nodes_temp.append((node, label - 1))
                else:
                    m_split_nodes_temp.append(nodes)

            m_cluster_label_max = m_cluster_label_max - 1

        append_node_i = node_i
        parent_i = self._parents[append_node_i]
        m_nodes.append([append_node_i, parent_i])
        while parent_i != ace_node:
            append_node_i = parent_i
            parent_i = self._parents[append_node_i]
            m_nodes.append([append_node_i, parent_i])
        append_node_j = node_j
        parent_j = self._parents[append_node_j]
        m_nodes.append([append_node_j, parent_j])
        while parent_j != ace_node:
            append_node_j = parent_j
            parent_j = self._parents[append_node_j]
            m_nodes.append([append_node_j, parent_j])

        return m_clustering, m_split_nodes_temp, m_cluster_label_max, m_nodes

    def refine(self, alpha=None, beta=None, runtime_id=0):
        alpha_pre_refine = self.alpha
        beta_pre_refine = self.beta
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.runtime = RUNTIME_OPTIONS[runtime_id]
        hash_string, previously_calculated = self.check_cache(alpha_pre_refine=alpha_pre_refine,
                                                              beta_pre_refine=beta_pre_refine)
        if previously_calculated is None:
            print("refine start ... ", end='')
            self._res_in_brief = ''
            self._iterate_refine()
            print("done")
            self.create_cache_version(hash_string)
        print(f'-------------------- ExClus - Dataset: {self.name} Emb: {self.emb_name} Alpha: {self.alpha} Beta: {self.beta} Ref. Runtime: {self.runtime}')
        print("Clusters: ", len(set(self._clustering_opt)))
        clusters = list(range(self._clusterlabel_max + 1))
        column_names = self.data.columns
        for i in clusters:
            print(f"  cluster {i}: {sum(self._clustering_opt == i)} points")
            print(f'    attributes: ', end='')
            for j in self._attributes_opt[i]:
                print(f'{column_names[j]} ', end='')
            print("")
        print("SI: ", self._si_opt)
        return self._clustering_opt, self._attributes_opt, self._si_opt

    def get_opt_values(self):
        return self._clustering_opt, self._attributes_opt, self._si_opt

    def get_ic_opt(self):
        return self._ic_opt

    def get_total_ic_opt(self):
        return self._total_ic_opt

    def get_priors(self):
        return self._priors

    def get_dls(self):
        return self._dls

    def save_adata(self, data_folder='.'):
        import anndata as ad
        file_name = data_folder + '/' + self.name + '.h5ad'
        adata = ad.read_h5ad(file_name)
        column_names = self.data.columns.tolist()
        # save ExClus information into .h5ad file
        adata.uns['ExClus'] = {}

        try:
            means = self.data.mean()
            stds = self.data.std()
        except TypeError:
            self.data = self.data.apply(lambda col: pd.factorize(col)[0])
            means = self.data.mean()
            stds = self.data.std()

        data_prior = np.hstack((means.values.reshape(len(means),1),stds.values.reshape(len(stds),1)))
        prior = pd.DataFrame(data_prior,  index=column_names, columns=['mean', 'var'])
        adata.uns['ExClus'] = {'si': self._si_opt, 'total-ic': self._total_ic_opt, 'priors': prior}
        for cluster in range(self._clusterlabel_max + 1):
            adata.uns['ExClus'][f'cluster {cluster}'] = {}
            adata.uns['ExClus'][f'cluster {cluster}']['attributes'] = [column_names[attr] for attr in self._attributes_opt[cluster]]
            adata.uns['ExClus'][f'cluster {cluster}']['ic'] = self._ic_opt[cluster]
        adata.obs['exclus-clustering'] = self._clustering_opt
        adata.write(file_name)
        return adata
