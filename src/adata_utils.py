import pandas as pd
import os
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE
from tsne import compute_tsne_series
# from umap import UMAP

VAR_TPYE_THRESHOLD = 20
VAR_TYPR_DICT = {'numeric': 1, 'categorical': 2}

# def get_var_complexity(var: pd.DataFrame) -> pd.Series:
#     '''
#     Get the complexity of the variable based on the type of the variable
#     :param var_type: the type of the variable
#     :return: the complexity of the variable
#     '''
#     # if the variable is numeric, it is considered 2, if it is categorical, it is considered as the count of distinct values
#     return var['var_type'].apply(lambda x: 2 if x == 'numeric' else len(var['var_type'].unique()))

# def get_var_type_complexity(data: pd.DataFrame) -> pd.DataFrame:
#     data_var_type_complexity = pd.DataFrame(columns=['var_type', 'var_complexity'])
#     if 'var_type' in data.columns:
#         if type(data['var_type']) == object:
#             data_var_type_complexity['var_type'] = data['var_type'].map(VAR_TYPR_DICT)
#         if type(data['var_complexity']) == int:
#             data_var_type_complexity['var_type'] = data['var_type']
#         for col_idx, var_type in enumerate(data['var_type']):
#             if var_type == 'numeric':
#                 data_var_type_complexity.loc[col_idx, 'var_complexity'] = 2
#             elif var_type == 'categorical':
#                 # 计算第 col_idx 列的不同取值数量
#                 column_data = data.iloc[:, col_idx]
#                 data_var_type_complexity.loc[col_idx, 'var_complexity'] = column_data.nunique()
#             else:
#                 print('ERROR! Unknown var_type {}'.format(var_type))
#                 data_var_type_complexity.loc[col_idx, 'var_complexity'] = None
#     else:
#         for col_idx, col_name in enumerate(data.columns):
#             col_data = data.iloc[:, col_idx]
#             distinct_counts = col_data.nunique()
#             if distinct_counts > VAR_TPYE_THRESHOLD:
#                 data_var_type_complexity.loc[col_idx, 'var_type'] = VAR_TYPR_DICT['numeric']
#                 data_var_type_complexity.loc[col_idx, 'var_complexity'] = 2
#             else:
#                 data_var_type_complexity.loc[col_idx, 'var_type'] = VAR_TYPR_DICT['categorical']
#                 data_var_type_complexity.loc[col_idx, 'var_complexity'] = distinct_counts
#     return data_var_type_complexity

# def get_scaled_data(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Preprocesses the input DataFrame:
#     - Standardizes numeric columns using StandardScaler.
#     - Encodes categorical columns (string type) into integers using factorize,
#       and then applies StandardScaler to the encoded values.
#
#     Parameters:
#         data (pd.DataFrame): Input data containing numeric and categorical columns.
#
#     Returns:
#         pd.DataFrame: Transformed data with standardized numeric columns
#                       and scaled categorical columns.
#     """
#     # Initialize an empty DataFrame to store processed data
#     processed_data = pd.DataFrame()
#
#     for col in data.columns:
#         if data[col].dtype == 'object':  # Check if the column is of string type
#             # Use factorize to map strings to unique integers
#             data[col], _ = pd.factorize(data[col])
#             # Scale the encoded integers using StandardScaler
#             scaler = StandardScaler()
#             processed_data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
#         elif pd.api.types.is_numeric_dtype(data[col]):  # Check if the column is numeric
#             # Use StandardScaler to standardize numeric columns
#             scaler = StandardScaler()
#             processed_data[col] = scaler.fit_transform(data[[col]])
#         else:
#             # Raise an error for unsupported data types
#             raise ValueError(f"Unsupported data type in column {col}")
#
#     return data, processed_data


def generate_adata(dataset_folder: str, dataset_name: str):
    '''
    Generate an AnnData object and save it as an h5ad file. AnnData object is used widely in this project as a role of saving meta data and computed infoclus result.

    Datails: given dataset name, generating adata file contains
    1. versions of dataset, including raw and scaled
    2. feature types, like numeric or categorical
    3. feature complexities, basically the minimum satitics needed to leaen the feature (mean&var for Guassian)
    3. embeddings of tsne and PCA
    
    Note: to make this function work, you will need the directory structure to be the same as the one in the repo
    '''
    # Load data
    data_path = os.path.join(dataset_folder, f'{dataset_name}.csv')
    df_data = pd.read_csv(data_path)
    df_data, data_scaled = get_scaled_data(df_data)
    
    # generate adata
    adata = ad.AnnData(X = data_scaled)
    adata.layers['raw_data'] = df_data.values
    adata.layers['scaled_data'] = data_scaled
    adata.var_names = df_data.columns.astype(str)
    # check feature type
    type_complexity = get_var_type_complexity(df_data)
    adata.var['var_type'] = int(type_complexity['var_type'].values)
    adata.var['var_complexity'] = int(type_complexity['var_complexity'].values)
    # todo: check the difference of tsne & PCA between here and Edith used.
    tsne = TSNE(n_components=2)
    adata.obsm['tsne'] = tsne.fit_transform(data_scaled)
    pca = PCA(n_components=2)
    adata.obsm['pca'] = pca.fit_transform(data_scaled)
    # # TODO: fix umap to work successfully
    # save adata
    adata.write(os.path.join(dataset_folder, f'{dataset_name}.h5ad'))
    print(f"adata file is saved at {os.path.join(dataset_folder, f'{dataset_name}.h5ad')}")
    return adata
