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
        self.var_type = self.adata.var['var_type']

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

        

        # self.emb_name = emb_name
        # self.data = df
        # self.data_scaled = df_scaled
        # self.embedding = embedding
        # self._binaryTargetsLen = lenBinary
        # self._samplesWeight = sampleWeight
        # if self._binaryTargetsLen == None:
        #     self._allAttType = 'categorical'
        #     if self._allAttType == 'categorical':
        #         self._valuesOfAttributes = []
        #         self._maxValuesOfAttributes = 0
        #         self._fixedDl = 0
        # self.model = model
        # self.alpha = alpha
        # self.beta = beta
        # self.epsilon = 0.00001
        # self.min_att = min_att
        # self.max_att = max_att
        # self.runtime = RUNTIME_OPTIONS[runtime_id]
        # self._targetsLen = len(self.data.columns)
        # self.cache_path = work_folder