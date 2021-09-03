import numpy as np
import copy
import collections
from queue import Queue
import queue
import time
from hashlib import sha256
from os.path import join

from caching import from_cache, to_cache

from sklearn.cluster import AgglomerativeClustering

WORK_FOLDER = '.'
CACHE_FOLDER = 'cache'

RUNTIME_OPTIONS = [0.5, 1, 5, 10, 30, 60, 300, 600, 1800, 3600, np.inf]


def kl_gaussian(mean1, std1, mean2, std2, epsilon=0.00001):
    std1 += epsilon
    std2 += epsilon
    a = np.log(std2 / std1) if std2 != 0 else 0
    b = (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2)
    return a + b - 1 / 2


def kl_bernoulli(p, q, epsilon=0.00001):
    if q == 0:
        q += epsilon
    elif q == 1:
        q -= epsilon
    a = p * np.log(p / q) if p != 0 else 0
    b = (1 - p) * np.log((1 - p) / (1 - q)) if p != 1 else 0
    return a + b


class ExclusOptimiser:

    def __init__(self, df, df_scaled, embedding,
                 model=AgglomerativeClustering(linkage="single", distance_threshold=0, n_clusters=None), name=None,
                 alpha=250, beta=1.6, runtime_id=0):
        self.name = name
        self.data = df
        self.data_scaled = df_scaled
        self.embedding = embedding
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.runtime = RUNTIME_OPTIONS[runtime_id]
        self._calc_priors()
        self._fit_model()
        self._clustering_opt = None
        self._attributes_opt = None
        self._ic_opt = None
        self._nodes_opt = None
        self._si_opt = 0
        self._total_dl_opt = 0
        self._total_ic_opt = 0
        self._splits = {}
        self._merge_candidates = {}
        self._dormant_merge_candidates = {}

        self.cache_path = join(WORK_FOLDER, CACHE_FOLDER)

    def _calc_priors(self):
        self._priors = {}
        self._dls = []
        column_names = self.data.columns
        means_prior = np.mean(self.data, axis=0)
        stds_prior = np.std(self.data, axis=0)
        for i in range(len(self.data.columns)):
            col = self.data.iloc[:, i]
            # Bernoulli for binary
            n_1 = (col == 1).sum()
            if n_1 == np.count_nonzero(col):
                self._priors[column_names[i]] = [n_1 / col.size]
                self._dls.append(1)
            # Gaussian otherwise
            else:
                self._priors[column_names[i]] = [means_prior[i], stds_prior[i]]
                self._dls.append(2)

        # Order attribute indices per dl to use later in dl optimisation
        unique_dls = sorted(set(self._dls))
        # Attributes indices split per dl, used to split IC into submatrix and later to find IC value of attribute
        self._dl_indices = collections.OrderedDict()
        for dl in unique_dls:
            # Fill dl_indices for one dl value
            indices = [i for i, value in enumerate(self._dls) if value == dl]
            self._dl_indices[dl] = indices

    def _create_linkage(self):
        # Create linkage matrix
        # create the counts of samples under each node
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        self._linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts]).astype(int)

    def _fit_model(self):
        self.model.fit_predict(self.embedding)
        self._create_linkage()

    # Return IC as matrix (row = cluster, column = attribute)
    def ic(self, clustering):
        # Separate data into its clusters
        ics = []
        n_components = len(set(clustering))
        for i in range(n_components):
            cluster = self.data.iloc[np.nonzero(clustering == i)[0], :]
            ics.append(self.ic_one_cluster(cluster))
        return ics, sum(sum(ics, []))

    def ic_one_cluster(self, cluster):
        columns = self.data.columns
        means_cluster = np.mean(cluster, axis=0)
        stds_cluster = np.std(cluster, axis=0)
        n_samples = len(cluster)
        cluster_ic = []
        for j in range(len(means_cluster)):
            # Bernoulli
            if len(self._priors[columns[j]]) == 1:
                p = means_cluster[j]
                q = self._priors[columns[j]][0]
                # Bernoulli kl fails if p or q is 1 or 0
                ic = kl_bernoulli(p, q)
            # Gaussian
            else:
                ic = kl_gaussian(means_cluster[j], stds_cluster[j], self._priors[columns[j]][0],
                                 self._priors[columns[j]][1])
            cluster_ic.append(n_samples * ic)
        return cluster_ic

    def _init_optimal_attributes_dl(self, ics):
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
        to_check_first = np.zeros((len(ics), len(unique_dls), 3), dtype=np.int8)

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
            q = Queue()
            q.queue = queue.deque(to_add)
            ics_dl[key] = q

        # Add attributes such that each cluster has one attribute at least
        # Attributes used to explain each cluster (row = cluster)
        attributes_total = [[value[0]] for value in best_combination]

        return attributes_total, ic_attributes, dl, best_comb_val, ics_dl

    # Binary bias because DL for binary attribute is only one
    # How to tune for alpha as by lowering it, IC will always increase
    def calc_optimal_attributes_dl(self, ics):
        ics = np.array(ics)

        # Ensures each cluster has at least one attribute
        attributes_total, ic_attributes, dl, best_comb_val, ics_dl = self._init_optimal_attributes_dl(ics)

        # Optimise
        old_value = -1
        new_value = best_comb_val
        ic_temp = 0
        dl_temp = 0
        while new_value > old_value:
            # New becomes old
            old_value = new_value
            # Check passed so update attributes, ic, and total dl + remove chosen attribute from its queue
            if old_value != best_comb_val:
                attr = ics_dl[dl_temp].get()
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
                    test_att = value.queue[0]
                except:
                    continue
                ic_test = ics[test_att[0]][self._dl_indices[key][test_att[1]]]
                # Only check att with higher dl if ic higher
                if ic_test < ic_temp:
                    continue
                new_test = (ic_attributes + ic_test) / (self.alpha + (dl + key) ** self.beta)
                if new_test > new_temp:
                    new_temp = new_test
                    ic_temp = ic_test
                    dl_temp = key
            new_value = new_temp

        return attributes_total, ic_attributes, dl, old_value

    """
    # Split (sub)tree into two clusters based on split node (root), return list of indices for one of clusters, (depth first)
    def _split(self, split_n):
        n_samples = len(self.model.labels_)
        split_int = int(split_n)
        # Leaf node reached
        if split_int < n_samples:
            return [split_int]
        # Not a leaf node, split into left and right subtree
        new_root = self._linkage_matrix[split_int - n_samples]
        lhs = self._split(new_root[0])
        rhs = self._split(new_root[1])
        return lhs + rhs
    """

    def _split(self, split_n):
        n_samples = len(self.model.labels_)
        if split_n < n_samples:
            return [int(split_n)]
        indices = []
        split_ints = [split_n]
        while split_ints:
            check = int(split_ints[0])
            del split_ints[0]
            new_root = self._linkage_matrix[check - n_samples]
            if new_root[0] < n_samples:
                indices.append(int(new_root[0]))
            else:
                split_ints.append(new_root[0])
            if new_root[1] < n_samples:
                indices.append(int(new_root[1]))
            else:
                split_ints.append(new_root[1])
        return indices

    def _cluster_indices_split(self, root_node, pre_index=None):
        n_samples = len(self.model.labels_)
        indices = copy.deepcopy(pre_index)
        if pre_index is None:
            indices = np.zeros(n_samples, dtype=int)
        new_cluster = max(indices) + 1
        # Only need to lookup lhs as rhs is complementary
        # Performance enhancement 1: only calc split if necessary
        if root_node[0] in self._splits:
            to_change = self._splits[root_node[0]]
        else:
            to_change = self._split(root_node[0])
            self._splits[root_node[0]] = to_change
        old_cluster = indices[to_change[0]]
        indices[to_change] = new_cluster

        return indices, new_cluster, old_cluster

    def _choose_optimal_split(self, nodes, clustering=None, ic_temp=None):
        n_samples = len(self.model.labels_)
        largest_si = 0
        largest_attributes = None
        largest_idx = None
        largest_clustering = None
        largest_ic = None
        largest_dl = 0
        largest_ic_attributes = 0
        largest_before_split = None
        largest_parent = None
        # Check all possible nodes of linkage matrix tree to make split and select one with highest SI
        for node_idx, parent in nodes:
            node = self._linkage_matrix[node_idx]
            new_clustering, new_cluster, old_cluster = self._cluster_indices_split(node, pre_index=clustering)
            # Performance enhancement 2: Only calc IC if necessary
            before_split = None
            if clustering is None:
                ics, total = self.ic(new_clustering)
            else:
                ics = copy.deepcopy(ic_temp)
                idx_old = np.nonzero(new_clustering == old_cluster)[0]
                idx_new = np.nonzero(new_clustering == new_cluster)[0]
                old_cluster_data = self.data.iloc[idx_old, :]
                ics[old_cluster] = self.ic_one_cluster(old_cluster_data)
                new_cluster_data = self.data.iloc[idx_new, :]
                ics.append(self.ic_one_cluster(new_cluster_data))
                before_split = np.append(idx_old, idx_new)

            attributes, ic_attributes, dl, si_val = self.calc_optimal_attributes_dl(ics)
            if si_val > largest_si:
                largest_si = si_val
                largest_attributes = attributes
                largest_idx = node_idx
                largest_parent = parent
                largest_clustering = new_clustering
                largest_ic = ics
                largest_dl = dl
                largest_ic_attributes = ic_attributes
                largest_before_split = before_split

        nodes.remove((largest_idx, largest_parent))
        opt_node = self._linkage_matrix[largest_idx]
        # Performance enhancement: only add nodes that have splits with more than 1 sample
        if opt_node[0] >= n_samples:
            l_child = self._linkage_matrix[opt_node[0] - n_samples]
            if l_child[0] >= n_samples or l_child[1] >= n_samples:
                nodes.append((opt_node[0] - n_samples, largest_idx))
        if opt_node[1] >= n_samples:
            r_child = self._linkage_matrix[opt_node[1] - n_samples]
            if r_child[0] >= n_samples or r_child[1] >= n_samples:
                nodes.append((opt_node[1] - n_samples, largest_idx))

        return nodes, largest_clustering, largest_attributes, largest_si, largest_ic, largest_ic_attributes, largest_dl, [
            largest_idx, largest_before_split, largest_parent, 0]

    def _iterate_levels(self):
        # List of nodes in linkage matrix that can be split into two clusters
        nodes = [(len(self._linkage_matrix) - 1, None)]
        clustering_new = None
        ic_new = None
        local_optimum = False
        iterations = 0
        to_merge = {}
        to_dormant = []
        start = time.time()
        while nodes and (time.time() - start < self.runtime):
            iterations += 1
            # Opt node is used so that in refine clusters can be merged, it is a list
            # First element is id of node just split, so that the specific split can be merged (allowed merges are stored in a dict)
            # Second element are all indices that need to become one cluster
            # Third element is parent node, to ensure it can become an allowed merge operation when necessary
            # Fourth element is a count to know when it is allowed to become an allowed merge operation once stored in the dormant merge dict
            nodes, clustering_new, attributes_new, si_val_new, ic_new, ic_att_new, dl_new, opt_node = self._choose_optimal_split(
                nodes,
                clustering=clustering_new,
                ic_temp=ic_new)
            to_merge[opt_node[0]] = opt_node[1:]
            if opt_node[2] is not None:
                to_dormant.append(opt_node[2])
            if si_val_new > self._si_opt:
                if local_optimum:
                    print("Local opt")
                    print("Clusters: ", len(set(self._clustering_opt)))
                    print("SI: ", self._si_opt)
                self._clustering_opt = copy.deepcopy(clustering_new)
                self._attributes_opt = copy.deepcopy(attributes_new)
                self._si_opt = si_val_new
                self._ic_opt = copy.deepcopy(ic_new)
                self._total_dl_opt = dl_new
                self._total_ic_opt = ic_att_new
                self._nodes_opt = copy.deepcopy(nodes)
                # Ensures merging is possible later after split
                self._merge_candidates.update(to_merge)
                to_merge.clear()
                for key in to_dormant:
                    value = self._merge_candidates.pop(key, None)
                    if value is not None:
                        value[2] += 1
                        self._dormant_merge_candidates[key] = value
                    else:
                        value = self._dormant_merge_candidates[key]
                        value[2] += 1
                        self._dormant_merge_candidates[key] = value
                to_dormant = []

            else:
                local_optimum = True

        # Sometimes, between local optima, a subsolution might not be that good, refine checks if merging some clusters might be better
        old_runtime = self.runtime
        self.runtime = RUNTIME_OPTIONS[1]
        self._iterate_refine()
        self.runtime = old_runtime
        print("Iterations", iterations)
        print("Clusters: ", len(set(self._clustering_opt)))
        print("SI: ", self._si_opt)
        print("")

    def _merge(self, clustering, ic_temp, context):
        labels = copy.deepcopy(clustering)
        ic = copy.deepcopy(ic_temp)
        unique_labels_total = set(labels)
        max_og_label = max(unique_labels_total)
        to_change = context[0]
        unique_labels_merge = set(labels[to_change])
        min_label = min(unique_labels_merge)
        max_label = max(unique_labels_merge)
        labels[to_change] = min_label
        ic[min_label] = self.ic_one_cluster(self.data.iloc[to_change, :])
        for i in range(max_label + 1, max_og_label + 1):
            indices = np.nonzero(labels == i)[0]
            labels[indices] = i - 1
            ic[i - 1] = ic[i]
        ic = ic[:-1]
        return labels, ic

    def _choose_optimal_merge(self, clustering, ic):
        opt_si = 0
        opt_ic = None
        opt_dl = 0
        opt_ic_att = 0
        opt_attributes = None
        opt_clustering = None
        opt_merge_id = None
        for node_id, context in self._merge_candidates.items():
            new_clustering, new_ic = self._merge(clustering, ic, context)
            attributes, ic_attributes, dl, si_val = self.calc_optimal_attributes_dl(new_ic)
            if si_val > opt_si:
                opt_ic = new_ic
                opt_attributes = attributes
                opt_clustering = new_clustering
                opt_merge_id = node_id
                opt_si = si_val
                opt_dl = dl
                opt_ic_att = ic_attributes
        return opt_merge_id, opt_clustering, opt_attributes, opt_si, opt_ic, opt_ic_att, opt_dl

    def _iterate_refine(self):
        s_nodes = copy.deepcopy(self._nodes_opt)
        s_clustering = copy.deepcopy(self._clustering_opt)
        s_ic = copy.deepcopy(self._ic_opt)
        m_clustering = copy.deepcopy(self._clustering_opt)
        m_ic = copy.deepcopy(self._ic_opt)

        si_opt = self._total_ic_opt / (self.alpha + self._total_dl_opt ** self.beta)
        n_samples = len(self.model.labels_)

        disable_merge = False
        disable_split = False
        start = time.time()
        while time.time() - start < self.runtime:
            if not disable_merge:
                m_id, m_clustering, m_attributes, m_si, m_ic, m_ic_att, m_dl = self._choose_optimal_merge(m_clustering,
                                                                                                          m_ic)
            if not disable_split:
                s_nodes, s_clustering, s_attributes, s_si, s_ic, s_ic_att, s_dl, s_opt_node = self._choose_optimal_split(
                    s_nodes,
                    clustering=s_clustering,
                    ic_temp=s_ic)

            if not disable_split and s_si > si_opt and s_si > m_si:
                self._clustering_opt = copy.deepcopy(s_clustering)
                self._attributes_opt = s_attributes
                si_opt = s_si
                self._ic_opt = copy.deepcopy(s_ic)
                self._total_dl_opt = s_dl
                self._total_ic_opt = s_ic_att
                self._nodes_opt = copy.deepcopy(s_nodes)
                self._merge_candidates[s_opt_node[0]] = s_opt_node[1:]

                value = self._merge_candidates.pop(s_opt_node[2], None)
                if value is not None:
                    value[2] += 1
                    self._dormant_merge_candidates[s_opt_node[2]] = value
                else:
                    value = self._dormant_merge_candidates[s_opt_node[2]]
                    value[2] += 1
                    self._dormant_merge_candidates[s_opt_node[2]] = value
                disable_merge = True

            elif not disable_merge and m_si > si_opt:
                self._clustering_opt = copy.deepcopy(m_clustering)
                self._attributes_opt = m_attributes
                si_opt = m_si
                self._ic_opt = copy.deepcopy(m_ic)
                self._total_dl_opt = m_dl
                self._total_ic_opt = m_ic_att

                merged_context = self._merge_candidates.pop(m_id)
                parent = self._dormant_merge_candidates[merged_context[1]]
                parent[2] -= 1
                if parent[2] == 0:
                    self._merge_candidates[merged_context[1]] = parent
                    del self._dormant_merge_candidates[merged_context[1]]
                else:
                    self._dormant_merge_candidates[merged_context[1]] = parent

                merged_node = self._linkage_matrix[m_id]
                if merged_node[0] >= n_samples:
                    l_child = self._linkage_matrix[merged_node[0] - n_samples]
                    if l_child[0] >= n_samples or l_child[1] >= n_samples:
                        self._nodes_opt.remove((merged_node[0] - n_samples, m_id))
                if merged_node[1] >= n_samples:
                    r_child = self._linkage_matrix[merged_node[1] - n_samples]
                    if r_child[0] >= n_samples or r_child[1] >= n_samples:
                        self._nodes_opt.remove((merged_node[1] - n_samples, m_id))
                self._nodes_opt.append((m_id, merged_context[1]))
                disable_split = True
            else:
                break
        self._si_opt = si_opt

    def optimise(self, alpha=None, beta=None, runtime_id=0):
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.runtime = RUNTIME_OPTIONS[runtime_id]
        hash_string, previously_calculated = self.check_cache()
        if previously_calculated is None:
            self._si_opt = 0
            self._merge_candidates = {}
            self._dormant_merge_candidates = {}
            self._iterate_levels()
            self.create_cache_version(hash_string)
        return self._clustering_opt, self._attributes_opt, self._si_opt

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
            self._iterate_refine()
            self.create_cache_version(hash_string)
        return self._clustering_opt, self._attributes_opt, self._si_opt

    def check_cache(self, alpha_pre_refine=0, beta_pre_refine=0):
        to_hash = f'{self.name}{self.alpha}{self.beta}{self.runtime}{alpha_pre_refine}{beta_pre_refine}'
        hash_string = sha256(to_hash.encode('utf-8')).hexdigest()
        previously_calculated = from_cache(join(self.cache_path, hash_string))
        if previously_calculated is not None:
            print("From cache")
            self._clustering_opt = previously_calculated["clustering"]
            self._attributes_opt = previously_calculated["attributes"]
            self._si_opt = previously_calculated["si"]
            self._ic_opt = previously_calculated["ic"]
            self._nodes_opt = previously_calculated["nodes"]
            self._total_dl_opt = previously_calculated["total_dl"]
            self._total_ic_opt = previously_calculated["total_ic"]
            self._splits.update(previously_calculated["splits"])
            self._merge_candidates = previously_calculated["merge_candidates"]
            self._dormant_merge_candidates = previously_calculated["dormant_merge_candidates"]
        return hash_string, previously_calculated

    def create_cache_version(self, hash_string):
        previously_calculated = {"clustering": self._clustering_opt, "attributes": self._attributes_opt,
                                 "si": self._si_opt, "ic": self._ic_opt, "nodes": self._nodes_opt,
                                 "total_dl": self._total_dl_opt, "total_ic": self._total_ic_opt,
                                 "splits": self._splits, "merge_candidates": self._merge_candidates,
                                 "dormant_merge_candidates": self._dormant_merge_candidates}
        to_cache(join(self.cache_path, hash_string), previously_calculated)

    def set_cache_path(self, path):
        self.cache_path = path

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
