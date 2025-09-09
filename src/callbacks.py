import json
import os

import dash, time
from dash.dependencies import Input, Output, State

from layout import *
from dash_utils import build_infoclus, serialize_obj, deserialize_obj
from infoclus_utils import get_opt_attributes, ic_one_info
from config import PROJECT_ROOT

def register_callbacks(app):

    @app.callback(
        Output('infoclus_store', 'data'),
        Output('dataset_store', 'data'),
        Output('embedding_store', 'data'),
        Output('clustering_store', 'data'),
        Output('clustering-to-show-select', 'options'),
        [Input('dataset-select', 'value'),
         Input('recalc-hyperparameters', 'value'),
         Input('import-labels', 'contents'),
         Input('import-labels', 'filename')
         ],
        [State('clustering_store', 'data'),
        State("embedding-select", "value"),
        State("alpha-slider", "value"),
        State("beta-slider", "value"),
        State("min-att-input", "value"),
        State("max-att-input", "value")]
    )
    def update_store(dataset, recalc_hyperparams, contents, filename, labels, embedding_name, alpha, beta, min_att, max_att):

        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        info_cache_update = None
        data_update = None
        embeddings_update = None
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'dataset-select':
            infoclus_obj = build_infoclus(dataset)
            info_cache_update = infoclus_obj.optimise()

            df_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', dataset, f'{dataset}.csv'))
            data_update = serialize_obj(df_data)

            embeddings_load = np.load(os.path.join(PROJECT_ROOT, 'data', dataset, 'cache', 'embeddings.npz'))
            embeddings = {k: embeddings_load[k] for k in embeddings_load.files}
            embeddings_update = serialize_obj(embeddings)
            labels = {'infoclus_clustering': info_cache_update['clustering']}
            options = dash.no_update

        elif trigger_id == 'recalc-hyperparameters':
            infoclus_obj = build_infoclus(dataset_name=dataset, emb_name=embedding_name)
            info_cache_update = infoclus_obj.optimise(alpha=alpha, beta=beta, min_att=min_att,max_att=max_att, run_id=int(recalc_hyperparams))
            data_update = dash.no_update
            embeddings_update = dash.no_update
            labels['infoclus_clustering'] = info_cache_update['clustering']
            options = dash.no_update

        elif trigger_id == 'import-labels':
            info_cache_update = dash.no_update
            data_update = dash.no_update
            embeddings_update = dash.no_update
            labels.update({filename: get_labels_from_input(contents)})
            options = []
            for key in labels.keys():
                options.append({'label': key, 'value': key})

        else:
            print('unknown trigger id')

        return info_cache_update, data_update, embeddings_update, labels, options


    @app.callback(
        Output('dashboard-content', 'children'),
        Input('infoclus_store', 'data'),
        [State('dataset_store', 'data'),
         State('embedding_store', 'data'),
         State('clustering_store', 'data'),]
    )
    def update_content(infoc_store, data_store, embedding_store, labels):
        infoc_dict = infoc_store
        df_data = deserialize_obj(data_store)
        embeddings_dict = deserialize_obj(embedding_store)
        return config_layout(infoc_para_res=infoc_dict, df_data=df_data, embeddings=embeddings_dict, labels=labels)


    @app.callback(
        Output('explanation', 'children'),
        Input('cluster-select', 'value'),
        [State('infoclus_store', 'data'),
         State('dataset_store', 'data'),]

    )
    def select_cluster_explanation(cluster_id, infoc_dict, data_store):
        df_data = deserialize_obj(data_store)
        return config_explanations(infoc_para_res=infoc_dict, df_data=df_data, cluster_label=cluster_id)

    @app.callback(
        Output('embedding-scatterPlot', 'figure'),
        [Input('embedding-for-show', 'value'),
        Input('clustering-to-show-select', 'value')],
        [State('clustering_store', 'data'),
         State('embedding_store', 'data')]
    )
    def select_embedding(emb_name,label_key, labels, embedding_store):

        embeddings_dict = deserialize_obj(embedding_store)

        return config_scatter_graph(labels[label_key],embeddings_dict[emb_name])

    @app.callback(
        Output('selected-explanation', 'children'),
        Input('embedding-scatterPlot', 'selectedData'),
        [
            State('infoclus_store', 'data'),
            State('dataset_store', 'data'),
        ]
    )
    def select_embedding(selected, infoc_dict, data_store):

        if selected is None:
            return "No points selected"
        if len(selected["points"]) == 0:
            return "No points selected"

        selected_idxes = [p["customdata"][0] for p in selected["points"]]
        scaled_data = np.array(infoc_dict['scaled_data'])
        selected_data = scaled_data[selected_idxes]
        mean_selected = np.mean(selected_data, axis=0)
        var_selected = np.var(selected_data, axis=0)
        count_selected = len(selected_idxes)
        prior = infoc_dict['prior']
        dls = infoc_dict['dls']
        ic_selected = ic_one_info(mean_selected, var_selected, count_selected, np.array(prior))
        attributes_total, ic_attributes, dl, best_comb_val = get_opt_attributes(alpha=infoc_dict['alpha'], beta=infoc_dict['beta'],dls=dls, ics=[ic_selected], min_att=infoc_dict['min_att'], max_att=infoc_dict['max_att'])

        return config_selected_explanations(infoc_para_res=infoc_dict, df_data=deserialize_obj(data_store), selected_idxes=selected_idxes,ics_cluster=ic_selected, attributes=attributes_total[0])

    @app.callback(
        Output('dataset-select', 'options'),
        Input('import-dataset', 'contents'),
        State('import-dataset', 'filename')
    )
    def import_dataset(contents, filename):
        if contents is None:
            return dash.no_update
        save_dataset_in_folder(contents, filename)
        return [{'label': dataset, 'value': dataset} for dataset in get_datasets()]

    # @app.callback(
    #     Output('clustering-to-show-select', 'options'),
    #     Output('clustering_store', 'data'),
    #     [Input('import-labels', 'contents'),
    #     Input('import-labels', 'filename')],
    #     [State('clustering_store', 'data'),]
    # )
    # def import_labels(contents, filename, labels):
    #     if contents is None:
    #         return dash.no_update, dash.no_update
    #
    #     labels.update({filename: get_labels_from_input(contents)})
    #     options = get_labels_from_input(labels)
    #     return options, labels



