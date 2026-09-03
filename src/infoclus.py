import os.path
import sys
import time, cProfile
from typing import Optional

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from caching import from_cache, to_cache
from infoclus_utils import *

from config import DATA_FOLDER, Random_State, REPLACE_NAN

RUNTIME_OPTIONS = [0.01, 0.5, 1, 5, 10, 30, 60, 180, 300, 600, 1800, 3600, np.inf]
VAR_TYPE_THRESHOLD = 0
EPSILON= 0.00001
KMEANS_COUNT = 30 # How many kmeans with different k we are going to consider, starting from the k passed in initialization
SPLITTING_STRATEGY = ['by_node','by_sibling']
COUNT_BASE_CLUSTERS = 200

class _Data:
    """
    Attributes:
        name
        dataset_folder
        cache_path
        data: pd.DataFrame
        data_raw: pd.DataFrame
        global_var_type
        var_type

        factorized_data: optional
        ls_mapping_chain_by_col: optional
    """
    def __init__(self, dataset_name: str, data: pd.DataFrame = None):
        self.name = dataset_name
        self.data: Optional[pd.DataFrame] = None
        self.data_raw: Optional[pd.DataFrame] = None
        self.prior = []

        self.dataset_folder = os.path.join(DATA_FOLDER, dataset_name)
        self.cache_path = os.path.join(self.dataset_folder, 'cache')

        if data is None:
            df_data = pd.read_csv(os.path.join(self.dataset_folder, f'{dataset_name}.csv'))
        else:
            df_data = data

        factorized_data, ls_mapping_chain_by_col, self.data , self.data_raw = get_scaled_data(df_data, REPLACE_NAN)
        self.size = self.data.shape[0]
        if factorized_data is not None and ls_mapping_chain_by_col is not None:
            self.factorized_data = factorized_data
            self.ls_mapping_chain_by_col = ls_mapping_chain_by_col

        df_var_type_complexity = get_var_type_complexity(self.data_raw, VAR_TYPE_THRESHOLD)
        self.var_type = df_var_type_complexity['var_type']
        if len(self.var_type.unique()) > 1:
            self.global_var_type = 'mixed'
        else:
            self.global_var_type = self.var_type.iloc[0]
        self._dls = list(df_var_type_complexity['var_complexity'])

class _Embeddings:
    """
    Attributes:
        all_embeddings
        embedding
    """
    def __init__(self, data: _Data, embedding: np.ndarray=None, emb_name=None):

        self.emb_name = None

        embeddings_path = os.path.join(data.dataset_folder, 'embeddings.npz')
        if os.path.exists(embeddings_path):
            print('Loading embeddings...')
            embeddings_load = np.load(embeddings_path)
            embeddings = {k: embeddings_load[k] for k in embeddings_load.files}
            if emb_name is not None and emb_name not in embeddings.keys():
                embeddings[emb_name] = embedding
            self.all_embeddings = embeddings
            print('Done')
        else:
            print('Creating embeddings files...')
            tic = time.time()
            if embedding is None:
                embeddings = get_embeddings(data.data.values)
            else:
                embeddings = {emb_name: embedding}
            self.all_embeddings = embeddings
            toc = time.time()
            print(f'Done, time: {toc - tic} s')
        np.savez(embeddings_path, **embeddings)

        if embedding is not None:
            if emb_name is None:
                print("Please give embedding name when embedding is given")
                sys.exit()
            self.embedding = embedding
            self.emb_name = emb_name
        else:
            self.emb_name = list(self.all_embeddings.keys())[0]
            self.embedding = self.all_embeddings[self.emb_name]


        # if embedding is None:
        #     if emb_name is None:
        #         self.emb_name = list(self.all_embeddings.keys())[0]
        #     else:
        #         self.emb_name = emb_name
        #     if self.emb_name not in self.all_embeddings.keys():
        #         print(f'Not supported embedding type {self.emb_name}, adopted to tSNE')
        #         self.emb_name = 'tsne'
        #     self.embedding = self.all_embeddings[self.emb_name]
        # else:
        #     self.embedding = embedding
        #     self.emb_name = emb_name
        #     np.savez(embeddings_path, **self.all_embeddings)

class _Model:
    """
    models
    Attributes:
        model

        meansForNodes
        varsForNodes
        nodesToPoints
        parents
    """
    def __init__(self):
        self.model = None

        self.linkage = None
        self.parents = {}
        self.meansForNodes = {}
        self.varsForNodes = {}
        self.nodesToPoints = {}

        self.modify_hierarchy = None
        self.base_clusters = None
        self.kmeans_model = None

        self.kmeans_mean = []
        self.kmeans_var = []
        self.kmeans_to_points = []

    def update_paras(self, modify_hierarchy: bool, linkage, emb_obj: _Embeddings, data_obj: _Data, base_clusters=None,):
        self.linkage = linkage
        self.model = AgglomerativeClustering(linkage=self.linkage, distance_threshold=0, n_clusters=None)
        self.modify_hierarchy = modify_hierarchy
        self.base_clusters = base_clusters

        if self.modify_hierarchy:
            if self.base_clusters is None:
                self.base_clusters = min(int(len(emb_obj.embedding) / 5), COUNT_BASE_CLUSTERS)
            else:
                self.base_clusters = base_clusters
            self.kmeans_model = KMeans(n_clusters=self.base_clusters, random_state=Random_State)
            self.kmeans_model.fit(emb_obj.embedding)
            self._compute_kmeans_statistics(data_obj.data.values)
            emb_obj.embedding = self.kmeans_model.cluster_centers_
        else:
            self.base_clusters = data_obj.size
        self.model.fit(emb_obj.embedding)
        self._calc_statistics_numeric(data_obj, self.modify_hierarchy)
        self._record_parents()

    def _compute_kmeans_statistics(self, data: np.ndarray):
        for cluster_label in range(self.base_clusters):
            set_of_samples = np.where(self.kmeans_model.labels_==cluster_label)[0]
            cluster = data[set_of_samples]
            self.kmeans_mean.append(np.mean(cluster,axis=0))
            self.kmeans_var.append(np.var(cluster, axis=0))
            self.kmeans_to_points.append(set_of_samples)

    def _calc_statistics_numeric(self, data_obj: _Data, modify_hierarchy: bool):

        data_value = data_obj.data.values
        n_samples = self.base_clusters
        for i, merge in enumerate(self.model.children_):

            self.nodesToPoints[i + n_samples] = []

            for j, node in enumerate(merge):
                if node < n_samples:
                    if modify_hierarchy:
                        self.meansForNodes[node] = self.kmeans_mean[node]
                        self.varsForNodes[node] = self.kmeans_var[node]
                        self.nodesToPoints[node] = self.kmeans_to_points[node]
                    else:
                        self.meansForNodes[node] = data_value[node]
                        self.varsForNodes[node] = np.zeros_like(self.meansForNodes[node])
                        self.nodesToPoints[node] = [node]
                self.nodesToPoints[i + n_samples].extend(self.nodesToPoints[node])

            self.meansForNodes[i + n_samples] = recur_mean(self.meansForNodes[merge[0]], len(self.nodesToPoints[merge[0]]),
                                                     self.meansForNodes[merge[1]], len(self.nodesToPoints[merge[1]]))
            self.varsForNodes[i + n_samples] = recur_var(self.meansForNodes[merge[0]],
                                                               self.varsForNodes[merge[0]],
                                                               len(self.nodesToPoints[merge[0]]),
                                                               self.meansForNodes[merge[1]],
                                                               self.varsForNodes[merge[1]],
                                                               len(self.nodesToPoints[merge[1]])
                                                               )

        data_obj.prior = [self.meansForNodes.get(self.base_clusters * 2 - 2),
                          self.varsForNodes.get(self.base_clusters * 2 - 2)]

    def _record_parents(self):

        for index, children in enumerate(self.model.children_):
            left_child = children[0]
            right_child = children[1]
            self.parents[left_child] = index + self.base_clusters
            self.parents[right_child] = index + self.base_clusters

    def get_ancestors(self, node_idx):
        node_ancestors_idxes = []
        child = node_idx
        parent = self.parents[child]
        # while self.parents.keys().__contains__(child):
        while child in self.parents:
            node_ancestors_idxes.append(self.parents[child])
            child = parent
            # if self.parents.keys().__contains__(child):
            if child in self.parents:
                parent = self.parents[child]
            else:
                break
        return node_ancestors_idxes

    def find_closest_ancestor(self, node_ancestor_idxes, candidate_ancestors_with_labels):

        closest_ancestor = None
        closest_ancestor_cluster_label = None
        for index, ancestor_info in enumerate(candidate_ancestors_with_labels):
            ancestor_cluster_label = index
            ancestor_node_idx = ancestor_info[0]
            ancestor_ancestors = ancestor_info[1]
            if ancestor_node_idx in node_ancestor_idxes:
                if closest_ancestor is None:
                    closest_ancestor = ancestor_node_idx
                    closest_ancestor_cluster_label = ancestor_cluster_label
                elif closest_ancestor in ancestor_ancestors:
                    closest_ancestor = ancestor_node_idx
                    closest_ancestor_cluster_label = ancestor_cluster_label

        return closest_ancestor, closest_ancestor_cluster_label

class _Result:
    def __init__(self, mean_prior: np.ndarray, var_prior: np.ndarray, base_clusters: int, data_size: int):

        self.iterations = 0

        self.ic_opt = [[0]*len(mean_prior)]  # ic of all attributes for each cluster
        self.si_opt = 0  # value of si for this clustering
        self.clusters_idxes_opt = [[i for i in range(data_size)]]  # all points belong to cluster 0
        self.attributes_opt = []  # chosen attributes for each cluster
        self.clusters_related_statistics_opt = [[mean_prior, var_prior, data_size]]
        self.split_nodes_opt = [[base_clusters * 2 - 2, []]]

        self.clustering = None
        self.count_clusters = None
        self.ic_val_per_cluster = None

    def update(self, ic_new, si_new, clusters_idxes_new, attributes_new, clusters_related_statistics_new, split_nodes_new):
        self.ic_opt = ic_new
        self.si_opt = si_new
        self.clusters_idxes_opt = clusters_idxes_new
        self.attributes_opt = attributes_new
        self.clusters_related_statistics_opt = clusters_related_statistics_new
        self.split_nodes_opt = split_nodes_new

    def _extend_results(self):

        clustering = np.full(sum(len(points) for points in self.clusters_idxes_opt), -1)
        for i, points in enumerate(self.clusters_idxes_opt):
            clustering[points] = i
        self.clustering = clustering
        self.count_clusters = len(self.clusters_idxes_opt)
        self.ic_val_per_cluster = [0]*self.count_clusters
        for i in range(self.count_clusters):
            self.ic_val_per_cluster[i] = sum(self.ic_opt[i][j] for j in self.attributes_opt[i])

    def get_info_results(self):
        self._extend_results()
        res={
            'clustering': self.clustering,
            'clusters_idxes_opt': self.clusters_idxes_opt,
            'count_clusters': self.count_clusters,
            'si_opt': self.si_opt,
            'ic_opt': self.ic_opt,
            'ic_val_per_cluster': self.ic_val_per_cluster,
            'attributes_opt': self.attributes_opt,
            'statistics': self.clusters_related_statistics_opt,
            'split_nodes_opt': self.split_nodes_opt
        }
        return res

class InfoClus:

    def __init__(self, dataset_name: str,     # necessary
                 data: pd.DataFrame = None,
                 embedding: np.ndarray = None,  # optional: given a precomputed embedding
                 emb_name=None,

                 linkage='single',
                 modify_hierarchical=True, base_clusters=None,

                 allow_cache = True
                 ):

        print('Initializing InfoClus ...')
        tic_initialization = time.time()

        self.epsilon = EPSILON
        self.allow_cache = allow_cache

        self.data_obj = _Data(dataset_name, data)
        self.embeddings_obj = _Embeddings(self.data_obj, embedding, emb_name)
        self.model_obj = _Model()
        self.model_obj.update_paras(modify_hierarchical, linkage, self.embeddings_obj, self.data_obj, base_clusters)
        self.result_obj = None

        file_path = os.path.join(self.data_obj.cache_path, self.embeddings_obj.emb_name + '_' + linkage + '_' + 'modify_' + str(modify_hierarchical))
        if self.allow_cache:
            to_cache(file_path, self)
            print(f'instance saved to {file_path}')

        toc_initialization = time.time()
        print(f'Initialization done, time: {toc_initialization - tic_initialization} s')

    def _update_paras(self,  alpha, beta, min_att, max_att, run_id, split_strategy_id, iteration_limit):

        if alpha is None:
            self.alpha = int(self.data_obj.size / 10)
        else:
            self.alpha = alpha
        if beta is None:
            self.beta = 1.5
        else:
            self.beta = beta
        if min_att is None:
            self.min_att = 2
        else:
            self.min_att = min_att
        if max_att is None:
            self.max_att = 5
        else:
            self.max_att = max_att
        if run_id is None:
            self.runtime_id = 4
        else:
            self.runtime_id = run_id
        self.runtime = RUNTIME_OPTIONS[self.runtime_id]

        # stopping criterion: by runtime (default) or by fixed number of iterations
        if iteration_limit is None:
            # keep original behaviour: use runtime limit only
            self.iteration_limit = None
            self.use_iteration_limit = False
        else:
            # when iteration_limit is provided, switch to iteration-based stopping
            self.iteration_limit = iteration_limit
            self.use_iteration_limit = True

        if split_strategy_id is None:
            self.split_strategy_id = 0
        else:
            self.split_strategy_id = split_strategy_id
        self.split_strategy = SPLITTING_STRATEGY[self.split_strategy_id]

    def get_paras(self):
        paras_val = {
            'data_name': self.data_obj.name,
            'scaled_data': self.data_obj.data.values.tolist(),
            'dls': self.data_obj._dls,
            'prior': [self.data_obj.prior[0].tolist(), self.data_obj.prior[1].tolist()],
            'global_arr_type': self.data_obj.global_var_type,
            'emb_name': self.embeddings_obj.emb_name,
            'linkage': self.model_obj.linkage,
            'modify_hierarchy': self.model_obj.modify_hierarchy,
            'base_clusters': self.model_obj.base_clusters,
            'alpha': self.alpha,
            'beta': self.beta,
            'min_att': self.min_att,
            'max_att': self.max_att,
            'runtime_id': self.runtime_id,
            'runtime': self.runtime,
            'use_iteration_limit': getattr(self, 'use_iteration_limit', False),
            'iteration_limit': getattr(self, 'iteration_limit', None),
            'split_strategy': self.split_strategy,
        }
        return paras_val

    ######################################## step 2: optimise: either run InfoClus or read from cache ########################################
    def optimise(self,
                 alpha=None, beta=None, min_att=None, max_att=None,
                 run_id=None, split_strategy_id=None, iteration_limit=None,
                 allow_cache=True):
        """
        optimise result with current hyperparameters, the process is as follows:
        1. update hyperparameters of self
        2. check cache
        3. start clustering when no cache
        4. print the clustering result
        """
        # update hyperparameters of self
        self.allow_cache = allow_cache
        self._update_paras(alpha, beta, min_att, max_att, run_id, split_strategy_id, iteration_limit)
        cache_name, cache_dict = self.check_cache()
        # start clustering when no cache
        if cache_dict is None:
            self.result_obj = _Result(self.data_obj.prior[0], self.data_obj.prior[1], self.model_obj.base_clusters, self.data_obj.size)
            self._run_infoclus_agglomerative()
            cache_dict = self.create_cache_version(cache_name, self.allow_cache)
            self.print_result_in_terminal()
        else:
            print('from cache')
        return cache_dict


    def print_result_in_terminal(self):
        print(
            f'\nInfoClus - Dataset: {self.data_obj.name} Emb: {self.embeddings_obj.emb_name} Alpha: {self.alpha} Beta: {self.beta} Ref. Runtime: {self.runtime}')
        print(f'Count of Clusters: {len(self.result_obj.clusters_idxes_opt)}')
        for cluster_idx in range(len(self.result_obj.clusters_idxes_opt)):
            print(f"    cluster {cluster_idx}:")
            print(f'        count of points: {len(self.result_obj.clusters_idxes_opt[cluster_idx])}')
            print(f'        attributes: ', end='')
            for j in self.result_obj.attributes_opt[cluster_idx]:
                print(f'{self.data_obj.data.columns[j]} ', end='')
            print("")
        print("SI: ", self.result_obj.si_opt)

    ######################################## step 3: run InfoClus by agglomerative ########################################
    def _run_infoclus_agglomerative(self):
        """
        Here is the core part of Infoclus algorithm, the process is as follows:
        1. initialization of all result-related variables as None
        2. iteration preparation and start iteration of splitting
           - either limited by a reference runtime (default)
           - or limited by a fixed number of iterations when `iteration_limit` is given

        Note: one split means enumerating all possible splits(nodes) and choose the best one to split one cluster into two
        """

        splitting_strategy = self.split_strategy

        #################################### step2: iteration #########################################

        if splitting_strategy == 'by_node':

            res_obj_local_opt = _Result(self.data_obj.prior[0], self.data_obj.prior[1], self.model_obj.base_clusters, self.data_obj.size)

            if self.model_obj.modify_hierarchy:
                candidates_for_split = set(range(self.model_obj.base_clusters*2-2))
            else:
                candidates_for_split = set(range(self.data_obj.size, 2*self.data_obj.size-2))

            count_iterations = 0
            start = time.time()
            print("\nsplitting by nodes start ... ")
            while len(candidates_for_split) > 0:
                # choose stopping criterion: iterations or runtime
                if getattr(self, 'use_iteration_limit', False):
                    if self.iteration_limit is not None and count_iterations >= self.iteration_limit:
                        break
                else:
                    if time.time() - start >= self.runtime:
                        break

                count_iterations += 1
                self._choose_optimal_split_by_nodes(res_obj_local_opt, candidates_for_split)

                # ##########################################
                # res_obj_local_opt._extend_results()
                # embedding = self.embeddings_obj.all_embeddings.get(self.embeddings_obj.emb_name)
                # df = pd.DataFrame({
                #     'x': embedding[:, 0],  # X coordinates
                #     'y': embedding[:, 1],  # Y coordinates
                #     'class': pd.Categorical(res_obj_local_opt.clustering)  # Classifications
                # })
                #
                # fig, ax = plt.subplots(figsize=(6, 6))
                # scatter = ax.scatter(
                #     df['x'],
                #     df['y'],
                #     c=df['class'].cat.codes,
                #     cmap='tab10',  # 颜色映射
                #     alpha=0.8,
                #     s=20
                # )
                # ax.set_facecolor("none")
                # ax.axis("off")
                # handles, labels = scatter.legend_elements(prop="colors")
                # ax.legend(handles, df['class'].cat.categories, title="Class", loc="best")
                #
                # plt.tight_layout()
                # plt.show()
                # ###################################

                if res_obj_local_opt.si_opt > self.result_obj.si_opt:
                    self.result_obj.update(copy.deepcopy(res_obj_local_opt.ic_opt),
                                           copy.deepcopy(res_obj_local_opt.si_opt),
                                           copy.deepcopy(res_obj_local_opt.clusters_idxes_opt),
                                           copy.deepcopy(res_obj_local_opt.attributes_opt),
                                           copy.deepcopy(res_obj_local_opt.clusters_related_statistics_opt),
                                           copy.deepcopy(res_obj_local_opt.split_nodes_opt))
            self.result_obj.iterations  = count_iterations
            print(f"{count_iterations} iterations done.")

    def _choose_optimal_split_by_nodes(self, res_obj_local_opt: _Result, candidates_for_split):

        largest_si = -1
        largest_changed_old_cluster_label = None
        largest_clusters_idxes_to_change = []
        largest_ics = []
        largest_statistics_to_change = []
        largest_split_nodes_to_change = []
        largest_attributes = []
        largest_nodes_idx = None

        for node_idx in candidates_for_split.copy():

            res = self._split_by_node(node_idx, res_obj_local_opt.clusters_idxes_opt, res_obj_local_opt.ic_opt,
                                      res_obj_local_opt.split_nodes_opt, res_obj_local_opt.clusters_related_statistics_opt)
            if res is None:
                candidates_for_split.remove(node_idx)
                continue

            ic_matrix = create_new_list_by_updating(old_list=res_obj_local_opt.ic_opt, new_list_to_change={res[0]: res[2][0]},
                                                    new_list_to_add=[res[2][1]])
            attributes, ic_attributes, dl, si = self.calc_optimal_attributes_dl(ic_matrix)

            if si > largest_si:
                largest_si = si
                largest_changed_old_cluster_label = res[0]
                largest_clusters_idxes_to_change = res[1]
                largest_ics = ic_matrix
                largest_split_nodes_to_change = res[3]
                largest_statistics_to_change = res[4]
                largest_attributes = attributes
                largest_nodes_idx = node_idx

        if len(candidates_for_split) == 0:
            return
        candidates_for_split.remove(largest_nodes_idx)

        largest_clusters_idxes = create_new_list_by_updating(old_list=res_obj_local_opt.clusters_idxes_opt,
                                                new_list_to_change={largest_changed_old_cluster_label: largest_clusters_idxes_to_change[0]},
                                                new_list_to_add=[largest_clusters_idxes_to_change[1]])
        largest_statistics = create_new_list_by_updating(old_list=res_obj_local_opt.clusters_related_statistics_opt,
                                                new_list_to_change={largest_changed_old_cluster_label: largest_statistics_to_change[0]},
                                                new_list_to_add=[largest_statistics_to_change[1]])
        res_obj_local_opt.split_nodes_opt.append(largest_split_nodes_to_change)

        res_obj_local_opt.update(largest_ics, largest_si, largest_clusters_idxes,
                                 largest_attributes, largest_statistics, res_obj_local_opt.split_nodes_opt)

    def _split_by_node(self, node, clusters_idxes, ic_matrix, split_nodes, statistics):

        clusters_idxes_to_change = [[],[]]
        ic_matrix_to_change = [[],[]]
        split_nodes_to_change = []
        statistics_to_change = [[],[]]

        points_to_change = self.model_obj.nodesToPoints[node]
        stat_for_new_clus = [self.model_obj.meansForNodes[node],
             self.model_obj.varsForNodes[node],
             len(points_to_change)]

        for clus_idx, split_node in enumerate(split_nodes):

            previous_node_ancestors_indexes = split_node[1]

            if node in previous_node_ancestors_indexes:

                cluster_set = set(clusters_idxes[clus_idx])
                points_to_change = [point for point in points_to_change if point not in cluster_set]
                # points_to_change = [point for point in points_to_change if point not in clusters_idxes[clus_idx]]
                if len(points_to_change) == 0:
                    return None
                elif len(points_to_change) < 0 :
                    print('error')
                stat_for_new_clus = recur_meanVar_remove(
                    stat_for_new_clus[0], stat_for_new_clus[1],stat_for_new_clus[2],
                    statistics[clus_idx][0], statistics[clus_idx][1], statistics[clus_idx][2]
                )
        stat_for_new_clus[2] = len(points_to_change)
        statistics_to_change[1] = stat_for_new_clus
        clusters_idxes_to_change[1] = points_to_change

        node_ancestors_idxes = self.model_obj.get_ancestors(node)
        closest_ancestor, previous_cluster_label = self.model_obj.find_closest_ancestor(node_ancestors_idxes, split_nodes)

        if closest_ancestor is not None:
            last_cluster_set = set(points_to_change)
            previous_cluster_idxes = [point for point in clusters_idxes[previous_cluster_label] if
                                      point not in last_cluster_set]

            clusters_idxes_to_change[0] = previous_cluster_idxes

            if len(previous_cluster_idxes) == 0:
                return None
            elif len(previous_cluster_idxes) < 0:
                print('error')
            statistics_to_change[0] = recur_meanVar_remove(
                statistics[previous_cluster_label][0],
                statistics[previous_cluster_label][1],
                statistics[previous_cluster_label][2],
                stat_for_new_clus[0],
                stat_for_new_clus[1],
                stat_for_new_clus[2]
            )

        ic_matrix_to_change[0] = ic_one_info(statistics_to_change[0][0], statistics_to_change[0][1],
                                             statistics_to_change[0][2], self.data_obj.prior)
        ic_matrix_to_change[1] = ic_one_info(statistics_to_change[1][0], statistics_to_change[1][1],
                                             statistics_to_change[1][2], self.data_obj.prior)
        split_nodes_to_change = [node, node_ancestors_idxes]

        return [previous_cluster_label, clusters_idxes_to_change, ic_matrix_to_change,
                split_nodes_to_change, statistics_to_change]

    def _init_optimal_attributes_dl(self, ics):

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
            dl = dl + sum((self.data_obj._dls[attribute]) for attribute in attributes)
            sortedic = np.delete(sortedic, index, axis=0)
            find_index = np.delete(find_index, index, axis=0)
        best_comb_val = ic_attributes / (self.alpha + dl ** self.beta)

        return attributes_total, ic_attributes, dl, best_comb_val, sortedic

    def calc_optimal_attributes_dl(self, ics):
        """
        This is a function used to return optimal attributes for each cluster.
        param
            ics: information content matrix (n*m), where n is count of clusters and m is the number of attributes.
        return
             attributes set for each cluster
        """
        ics = np.array(ics)
        attributes_total, ic_attributes, dl, best_comb_val, sortedic = self._init_optimal_attributes_dl(ics)
        out_max_att_limit = False
        while not out_max_att_limit and len(sortedic) > 0:
            extend_cluster_try = sortedic[0][0]
            extend_attr_try = sortedic[0][1]
            sortedic = np.delete(sortedic, 0, axis=0)
            if len(attributes_total[extend_cluster_try]) >= self.max_att:
                continue
            dl_try = dl + self.data_obj._dls[extend_attr_try]
            ic_attributes_try = ic_attributes + ics[extend_cluster_try, extend_attr_try]
            si_try = ic_attributes_try / (self.alpha + dl_try ** self.beta)
            if si_try >= best_comb_val:
                best_comb_val = si_try
                attributes_total[extend_cluster_try].append(extend_attr_try)
                dl = dl_try
                ic_attributes = ic_attributes_try
                out_max_att_limit = all(len(attribute) >= self.max_att for attribute in attributes_total)
            else:
                break

        return attributes_total, ic_attributes, dl, best_comb_val

    def create_cache_version(self, cache_name, allow_cache=True):
        info_paras_dict = self.get_paras()
        res_dict = self.result_obj.get_info_results()
        cache_dict = info_paras_dict | res_dict
        if allow_cache:
            to_cache(os.path.join(self.data_obj.cache_path, cache_name), cache_dict)
        return cache_dict

    def check_cache(self):

        current_paras = self.get_paras()
        cache_name = get_hashkey_from_dict(current_paras)

        pre_calc = from_cache(os.path.join(self.data_obj.cache_path, cache_name))
        return cache_name, pre_calc

    # def compute_explanation_given_clustering(self, clustering: np.ndarray, data: pd.DataFrame):
    #
    #     index_dict = {}
    #     for i, val in enumerate(clustering):
    #         index_dict.setdefault(val, []).append(i)
    #     ics = []
    #     prior = [np.mean(data, axis=0), np.var(data, axis=0)]
    #     for cluster_label in range(len(set(clustering))):
    #         index_cluster = index_dict[cluster_label]
    #         cluster = data.values[index_cluster]
    #         mean_cluster = np.mean(cluster, axis=0)
    #         var_cluster = np.var(cluster, axis=0)
    #         count_cluster = len(cluster)
    #         ic_cluster = ic_one_info(mean_cluster, var_cluster, count_cluster, prior)
    #         ics.append(ic_cluster)
    #     attributes, ic_attributes, dl, si_val = self._calc_optimal_attributes_dl(ics)
    #
    #     return attributes, si_val