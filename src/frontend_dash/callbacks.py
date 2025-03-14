import os
import dash
import pdb


from dash.dependencies import Input, Output, State
from layout import *
from src.frontend_dash.layout import ROOT_DIR
from utils import *
from src.utils import load_data, get_project_root
from src.infoclus import InfoClus

ROOT_DIR = get_project_root()

def register_callbacks(app):
    @app.callback(
        Output('main-div', 'children'),
        [
            Input('dataset-select', 'value'),
            Input("recalc-hyperparameters", "n_clicks")
        ],
        [
         State("embedding-select", "value"),
         State("alpha-slider", "value"),
         State("beta-slider", "value"),
         State("runtime-slider", "value"),
         State("minAtt-slider", "value")],
        prevent_initial_call=True
    )
    def update_maindiv(dataset_name, change_hyper, embedding_name, alpha, beta, runtime_id, minAtt):
        ctx = dash.callback_context  # 获取当前触发源
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'dataset-select':
            return config_layout(dataset_name=dataset_name, emb_name='tsne')
        else:
            # todo: add maxAtt into function
            infoclus_object_path = os.path.join(ROOT_DIR,'data', dataset_name, f'{dataset_name}_{embedding_name}.pkl')
            if_exists = os.path.exists(infoclus_object_path)
            if if_exists:
                with open(os.path.join(infoclus_object_path), 'rb') as f:
                    infoclus = pickle.load(f)
            else:
                infoclus = InfoClus(dataset_name=dataset_name, main_emb=embedding_name)
            infoclus.optimise(alpha=alpha,beta=beta,min_att=minAtt,runtime_id=runtime_id)
            return config_layout(dataset_name=dataset_name, emb_name=embedding_name)

    @app.callback(
        Output('explanation', 'children'),
        Input('cluster-select', 'value'),
        State('dataset-select', 'value'),
        State('embedding-used-for-clustering', 'value'),
        prevent_initial_call=True
    )
    def select_cluster_explanation(cluster_id, dataset_name, emb_name):
        file_path = os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}_{emb_name}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                infoclus = pickle.load(f)

            data = infoclus.data
            clustering = infoclus._clustering_opt
            if clustering is None or len(clustering) == 0:
                print('error, callbacks.py - line 65')
            att_names = infoclus.data_raw.columns.values
            attributes = infoclus._attributes_opt[cluster_id]
            ics_cluster = np.array(infoclus._ic_opt[cluster_id])
        else:
            print(f'Error, {dataset_name}_{emb_name}.pkl does not exist')
        return config_explanations(infoclus, data, clustering, attributes, att_names, ics_cluster, cluster_id)

    # todo: remove all_duplicate, using dash.callback_context
    @app.callback(
        Output('embedding-scatterPlot', 'figure', allow_duplicate=True),
        Input('embedding-select', 'value'),
        State('dataset-select', 'value'),
        State('embedding-used-for-clustering', 'value'),
        prevent_initial_call=True
    )
    def select_embedding(embedding_name_show, dataset_name, emb_name_comp):

        file_path = os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}_{emb_name_comp}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                infoclus = pickle.load(f)
        else:
            print(f'Error, {dataset_name}_{emb_name_comp}.pkl does not exist')

        return config_scatter_graph(infoclus,embedding_name_show)