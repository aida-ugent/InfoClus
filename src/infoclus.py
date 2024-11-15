import pandas as pd
from sklearn.cluster import AgglomerativeClustering


class InfoClus:

    def __init__(self, name: str, data: pd.DataFrame,
                 main_emb: str = 'tSNE_1',
                 model = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters=None)
                 ):

        self.name = name
        self.data = data
        # todo: get scaled data, also dataframe
        # todo: get features types, and ifmixed
        self.emb_name = main_emb
        self.model = model

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