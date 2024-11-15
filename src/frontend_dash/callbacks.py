import pandas as pd
import anndata as ad

from dash.dependencies import Input, Output, State
from dicts2adata import dicts2adata
from layout import get_scatter, config_explanations_kde
from src.si import ExclusOptimiser, RUNTIME_OPTIONS
from src.utils import load_data


def register_callbacks(app):
    @app.callback(
        [Output("embedding-scatterPlot", "figure"),
         Output("cluster-select", "options"),
         Output("cluster-select", "value")],
        [Input("recalc-hyperparameters", "n_clicks")],
        [State("dataset-select", "value"),
         State("embedding-select", "value"),
         State("alpha-slider", "value"),
         State("beta-slider", "value"),
         State("runtime-slider", "value"),
         State("minAtt-slider", "value")],
        prevent_initial_call=True
    )
    def optimise_different_hyperparameters(change_hyper, dataset_name, embedding_name, alpha, beta, runtime_id, minAtt):
        # todo: get embedding, seperate initial embedding computaion from trace to here

        adata_path = f'{dataset_name}.h5ad'
        adata = ad.read_h5ad(adata_path)

        data = adata.layers['raw_data']
        df = pd.DataFrame(data, columns=adata.var.index)
        df_data, df_scaled, lenBinary = load_data(df)
        embedding = adata.obsm.get(embedding_name)
        # todo: max_att not be applied in si, remove it or revise it
        optimiser = ExclusOptimiser(df, df_scaled, lenBinary,
                                    embedding, name=dataset_name, emb_name=embedding_name,
                                    alpha=alpha, beta=beta, min_att=minAtt, max_att=0, runtime_id=runtime_id,
                                    work_folder=f'../../data/{dataset_name}')

        # todo: finally, integrate the adata write in into exclusoptimiser, or like Zander, first store in optimiser not another file
        dicts2adata(dataset_name, adata_path,
                    f'../../data/{dataset_name}/{dataset_name}-{embedding_name}-single-{alpha}-{beta}-{minAtt}-{int(RUNTIME_OPTIONS[runtime_id])}-0-0')
        updated_adata = ad.read_h5ad(adata_path)

        # todo: fix get_scatter, make stype consistent, reduce redundant
        return get_scatter(updated_adata.obs['infoclus_clustering'].values,
                                    updated_adata.obsm.get(embedding_name)), \
            [{'label': "Cluster " + str(i), 'value': i} for i in
             list(set(updated_adata.obs['infoclus_clustering'].values))], \
            0

    @app.callback(
        Output('explanation', 'children'),
        Input('cluster-select', 'value'),
        State('dataset-select', 'value'),
        prevent_initial_call=True
    )
    def select_cluster_explanation(cluster_id, dataset_name):
        adata = ad.read_h5ad(f'{dataset_name}.h5ad')

        data = adata.X
        clustering = adata.obs['infoclus_clustering'].values
        att_names = adata.var.index.values
        attributes = adata.uns['InfoClus'][f'cluster_{cluster_id}']['attributes']
        ics_cluster = adata.uns['InfoClus'][f'cluster_{cluster_id}']['ic']
        return config_explanations_kde(data, clustering, attributes, att_names, ics_cluster, cluster_id)

    # todo: remove all_duplicate, using dash.callback_context
    @app.callback(
        Output('embedding-scatterPlot', 'figure', allow_duplicate=True),
        Input('embedding-select', 'value'),
        State('dataset-select', 'value'),
        prevent_initial_call=True
    )
    def select_cluster_explanation(embedding_name, dataset_name):

        adata = ad.read_h5ad(f'{dataset_name}.h5ad')

        return get_scatter(adata.obs['infoclus_clustering'].values,
                                    adata.obsm.get(embedding_name))