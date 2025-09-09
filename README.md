## Requirements
* notebook
* scikit-learn
* plotly <=5.20.0 (selectedData attributes)

* dash
* dash-bootstrap-components
* pandas

## Run

```dash
python app.py
```

InfoClus Class:
- name: dataset_name
- data_raw: DataFrame, read from original .csv file without any process
- all_embeddings: Dict with format {'emb_method': embedding of data_raw}
- data: DataFrame, processed for computation
- emb_name: str: emb_method selected for computing
- embedding: np.ndarray, embedding get from method-emb_name for data
- global_var_type, _var_type

- linkage: {‘ward’, ‘complete’, ‘average’, ‘single’}, the linkage used in hierarchical clustering
- model: hierarchical clustering model used in InfoClus
- modify_hierarchical: bool, a signal for whether modifying hierarchical clustering based on k-means or not
- _base_clusters: if modify_hierarchical, how many clusters set for k-means
- _kmedoids_model, _kmedoids_clustering

- alpha
- beta
- epsilon
- min_att
- max_att
- runtime_id
- split_strategy

- allow_cache
- _cache_folder
- _dataset_folder

- attributes_opt
- clustering_opt
- clusters_idxes_opt
- ic_opt
- si_opt
- _clustersRelatedInfo
- _meansForNodes
- _nodestoPoints
- _linkage_matrix
- _parents
- _parents_of_all_nodes
- _priors

data folder:
- dataset_name
  - cache
    - dataset_name_modify_true # initialized InfoClus object with modified hierarchy
    - dataset_name_modify_false # initialized InfoClus object with modified hierarchy
    - others # InfoClus object running under different parameters

Analyze your dataset by InfoClus, assuming your data has name 'toy':
- process data
  - generate a folder with your dataset name 'toy' under \data folder, 
  - generate a folder named 'cache'
  - put your data, toy.csv, into 'toy' folder

