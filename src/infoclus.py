import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from adata_utils import generate_adata
import numpy as np
import os


RUNTIME_OPTIONS = [0.01, 0.5, 1, 5, 10, 30, 60, 180, 300, 600, 1800, 3600, np.inf]


class InfoClus:
    '''
        InfoClus is a class that offers a explanable clustering solution for datasets
        It uses a pre-clustering model to offer candidate clusters

        Attributes:
        name: name of the dataset
        relative_data_path: the path to the data folder
        emb_name: the embedding to be used for clustering
        model: the pre-clustering model to be used to offer candidate clusters

        adata: the adata object that contains the dataset and some related information

    '''

    def __init__(self, dataset_name: str, relative_data_path: str,
                 main_emb: str = 'tSNE_1',
                 model = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None),
                 alpha: int = None, beta: float = 1.5, min_att: int = 2, max_att: int = 10, runtime_id: int = 3
                 ):
        '''

        :param name: name of the dataset
        :param relative_data_path: the path to the data folder
        :param main_emb: the embedding to be used for clustering
        :param model: the pre-clustering model to be used to offer candidate clusters
        '''

        self.name = dataset_name
        self.relative_data_path = relative_data_path
        self.emb_name = main_emb
        self.model = model

        self.adata = generate_adata(self.relative_data_path, self.name)

        self.data_raw = pd.DataFrame(self.adata.layers['raw'], columns=self.adata.var_names)
        self.data_scaled = pd.DataFrame(self.adata.layers['scaled'], columns=self.adata.var_names)
        self.embedding = self.adata.obsm[self.emb_name]
        # TODO: check the type of self.var_type
        # TODO: change name of _dls to be more cohesive with the code
        self.var_type = self.adata.var['var_type']
        self._dls = self.adata.var['var_complexity']
        if alpha is None:
            self.alpha = int(self.adata.obs.size/10)
        else:
            self.alpha = alpha
        self.beta = beta
        self.min_att = min_att
        self.max_att = max_att
        self.runtime_id = runtime_id
        self.runtime = RUNTIME_OPTIONS[runtime_id]
        # TODO: check if self_cache_path is correct
        self.cache_path = os.path.join(self.relative_data_path, self.name, 'cache')

        # calculation
        self._fit_model()  # get agglomarative clustering and get linage matrix
        self._calc_priors()

        self._parents = None
        self._linkage_matrix = None
        self._statistics_for_nodes = None
        self._prior_dataset = None

    def _fit_model(self):
        '''
        Fit the pre-clustering model
        :return:
        '''
        self.model.fit(self.embedding)
        self._create_linkage()

    def _create_linkage(self):
        '''
        TODO: working... rewrite this function to be more clear
        TODO: check the model to see if _parents are needed
        Create linkage matrix and parents of nodes
        :return:
        '''
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
        # self._clusterTree_root = to_tree(self._linkage_matrix, rd=False)