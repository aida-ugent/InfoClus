import base64
import json
import os

import dash, time
from dash.dependencies import Input, Output, State, ALL

from layout import *
from dash_utils import build_infoclus, serialize_obj, deserialize_obj, create_prompt, query_llm, parse_llm_cluster_labels
from infoclus_utils import get_opt_attributes, ic_one_info
from config import PROJECT_ROOT

def register_callbacks(app):

    # @app.callback(
    #     Output('dashboard-content', 'children'),
    #     Input('dataset-select', 'value')
    # )
    # def update_layout(dataset):
    #
    #     infoclus_obj = build_infoclus(dataset)
    #     dict_infoRes = infoclus_obj.optimize()
    #
    #     df_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', dataset, f'{dataset}.csv'))
    #     embeddings_load = np.load(os.path.join(PROJECT_ROOT, 'data', dataset, 'embeddings.npz'))
    #     embeddings = {k: embeddings_load[k] for k in embeddings_load.files}
    #     labels = {'InfoClus': dict_infoRes['clustering']}
    #
    #     return config_layout(dict_infoRes, df_data, embeddings, labels, 0)

    @app.callback(
        Output('infoclus_store', 'data'),
        Output('dataset_store', 'data'),
        Output('embedding_store', 'data'),
        Output('clustering_store', 'data'),
        Output('clustering-dropdown-menu', 'children'),
        Output('clustering-value-store', 'data'),
        [Input('dataset-select', 'value'),
         Input('recalc-hyperparameters', 'value'),
         Input('import-labels', 'contents'),
         Input('import-labels', 'filename'),
         Input('import-embedding', 'contents'),
         Input('import-embedding', 'filename')
         ],
        [State('clustering_store', 'data'),
        State('clustering-value-store', 'data'),
        State("embedding-select", "value"),
        State("alpha-slider", "value"),
        State("beta-slider", "value"),
        State("min-att-input", "value"),
        State("max-att-input", "value")]
    )
    def update_store(dataset, recalc_hyperparams, labels_contents, labels_filename, embedding_contents, embedding_filename,
                     labels, current_clustering_value, embedding_name, alpha, beta, min_att, max_att):

        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        info_cache_update = None
        data_update = None
        embeddings_update = None
        menu_children_update = dash.no_update
        clustering_value_update = dash.no_update
        alpha = float(alpha) if alpha is not None else 1.0
        beta = float(beta) if beta is not None else 1.0
        min_att = int(min_att) if min_att is not None else 0
        max_att = int(max_att) if max_att is not None else 0

        trigger_id_raw = ctx.triggered[0]['prop_id']
        trigger_id = trigger_id_raw.split('.')[0] if isinstance(trigger_id_raw, str) else trigger_id_raw
        if trigger_id == 'dataset-select':
            infoclus_obj = build_infoclus(dataset)
            info_cache_update = infoclus_obj.optimise()

            df_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', dataset, f'{dataset}.csv'))
            data_update = serialize_obj(df_data)

            embeddings_load = np.load(os.path.join(PROJECT_ROOT, 'data', dataset, 'embeddings.npz'))
            embeddings = {k: embeddings_load[k] for k in embeddings_load.files}
            embeddings_update = serialize_obj(embeddings)
            labels = {'InfoClus': info_cache_update['clustering']}

        elif trigger_id == 'recalc-hyperparameters':
            infoclus_obj = build_infoclus(dataset_name=dataset, emb_name=embedding_name)
            info_cache_update = infoclus_obj.optimise(alpha=alpha, beta=beta, min_att=min_att,max_att=max_att, run_id=int(recalc_hyperparams))
            data_update = dash.no_update
            embeddings_update = dash.no_update
            labels['InfoClus'] = info_cache_update['clustering']

        elif trigger_id == 'import-labels':
            if labels_contents is None or labels_filename is None:
                return dash.no_update, dash.no_update, dash.no_update, labels, dash.no_update, dash.no_update

            info_cache_update = dash.no_update
            data_update = dash.no_update
            embeddings_update = dash.no_update
            labels.update({labels_filename: get_labels_from_input(labels_contents)})
            clustering_value_update = labels_filename
            menu_children_update = [
                dbc.DropdownMenuItem(k, id={'type': 'cluster-option', 'index': k})
                for k in labels.keys()
            ] + [
                dbc.DropdownMenuItem(
                    html.Div(
                        dcc.Upload(
                            id='import-labels',
                            children=['Upload labels...'],
                            style={'cursor': 'pointer', 'padding': '0.25rem 0.5rem'},
                        ),
                        style={'width': '100%'},
                    ),
                    id='cluster-option-upload',
                )
            ]

        elif trigger_id == 'import-embedding':

            embeddings_path = os.path.join(PROJECT_ROOT, 'data', dataset, 'cache', 'embeddings.npz')
            embeddings_load = np.load(embeddings_path)
            embeddings = {k: embeddings_load[k] for k in embeddings_load.files}

            embedding_contents_type, embedding_contents_string = embedding_contents.split(',')
            decoded_embedding = base64.b64decode(embedding_contents_string)
            embeddings[embedding_filename] = decoded_embedding
            np.savez(embeddings_path, **embeddings)
            embeddings_update = serialize_obj(embeddings)

            info_cache_update = dash.no_update
            data_update = dash.no_update
            labels = dash.no_update

        else:
            print('unknown trigger id')

        return info_cache_update, data_update, embeddings_update, labels, menu_children_update, clustering_value_update


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
        Output('clustering-value-store', 'data', allow_duplicate=True),
        Input({'type': 'cluster-option', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True,
    )
    def select_clustering_from_menu(n_clicks_list):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        triggered_id = ctx.triggered[0]['prop_id']
        if triggered_id == '.':
            raise dash.exceptions.PreventUpdate
        try:
            id_dict = json.loads(triggered_id.replace('.n_clicks', ''))
            if id_dict.get('type') == 'cluster-option':
                return id_dict['index']
        except Exception:
            pass
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output('clustering-dropdown-menu', 'label'),
        Input('clustering-value-store', 'data'),
    )
    def sync_clustering_menu_label(store_value):
        return store_value if store_value is not None else 'InfoClus'

    @app.callback(
        Output('embedding-select', 'options'),
        Output('embedding-dropdown-menu-show', 'children'),
        Input('embedding_store', 'data')
    )
    def update_options(embeddings_serialised):
        embeddings_dict = deserialize_obj(embeddings_serialised)
        options = [{'label': emb_name, 'value': emb_name} for emb_name in embeddings_dict.keys()]
        menu_children = [
            dbc.DropdownMenuItem(emb_name, id={'type': 'embedding-option', 'index': emb_name})
            for emb_name in embeddings_dict.keys()
        ]
        return options, menu_children

    @app.callback(
        Output('embedding-value-store', 'data'),
        Input({'type': 'embedding-option', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True,
    )
    def select_embedding_from_menu(n_clicks_list):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        triggered_id = ctx.triggered[0]['prop_id']
        if triggered_id == '.':
            raise dash.exceptions.PreventUpdate
        try:
            id_dict = json.loads(triggered_id.replace('.n_clicks', ''))
            if id_dict.get('type') == 'embedding-option':
                return id_dict['index']
        except Exception:
            pass
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output('embedding-dropdown-menu-show', 'label'),
        Input('embedding-value-store', 'data'),
    )
    def sync_embedding_menu_label(store_value):
        return store_value if store_value is not None else 'tsne'

    @app.callback(
        Output('cluster-select-value-store', 'data', allow_duplicate=True),
        Input({'type': 'cluster-select-option', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True,
    )
    def select_cluster_from_menu(n_clicks_list):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        triggered_id = ctx.triggered[0]['prop_id']
        if triggered_id == '.':
            raise dash.exceptions.PreventUpdate
        try:
            id_dict = json.loads(triggered_id.replace('.n_clicks', ''))
            if id_dict.get('type') == 'cluster-select-option':
                return id_dict['index']
        except Exception:
            pass
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output('cluster-select-menu', 'label'),
        Input('cluster-select-value-store', 'data'),
    )
    def sync_cluster_select_menu_label(store_value):
        if store_value is None:
            return 'Cluster 0'
        return f'Cluster {store_value}'

    @app.callback(
        Output('explanation', 'children'),
        Input('cluster-select-value-store', 'data'),
        [State('infoclus_store', 'data'),
         State('dataset_store', 'data'),]

    )
    def select_cluster_explanation(cluster_id, infoc_dict, data_store):
        df_data = deserialize_obj(data_store)
        return config_explanations(infoc_para_res=infoc_dict, df_data=df_data, cluster_label=cluster_id)

    @app.callback(
        Output('cluster-select-tooltip', 'children'),
        Input('cluster-select-value-store', 'data'),
        State('infoclus_store', 'data'),
    )
    def update_cluster_select_tooltip(cluster_id, infoc_dict):
        if infoc_dict is None or cluster_id is None:
            return "The cluster is generated by InfoClus and contains x% of data."
        scaled_data = infoc_dict.get('scaled_data')
        clusters_idxes = infoc_dict.get('clusters_idxes_opt', [])
        if not scaled_data or cluster_id >= len(clusters_idxes):
            return "The cluster is generated by InfoClus and contains x% of data."
        n_total = len(scaled_data)
        n_cluster = len(clusters_idxes[cluster_id])
        pct = (n_cluster / n_total * 100) if n_total else 0
        return f"The cluster is generated by InfoClus and contains {pct:.2f}% of data."

    @app.callback(
        Output('scatter-point-style-collapse', 'is_open'),
        Input('scatter-point-style-toggle', 'n_clicks'),
        State('scatter-point-style-collapse', 'is_open'),
        prevent_initial_call=True,
    )
    def toggle_scatter_point_style(n_clicks, is_open):
        return not is_open

    @app.callback(
        [Output('embedding-scatterPlot', 'figure'),
         Output('scatter-plot-title-container', 'children'),
         Output('scatter-plot-tooltip', 'children')],
        [Input('embedding-value-store', 'data'),
         Input('clustering-value-store', 'data'),
         Input('scatter-point-size', 'value'),
         Input('scatter-point-opacity', 'value')],
        [State('clustering_store', 'data'),
         State('embedding_store', 'data'),
         State('infoclus_store', 'data')]
    )
    def update_scatter_plot(emb_name, label_key, point_size, point_opacity, labels, embedding_store, infoc_store):

        embeddings_dict = deserialize_obj(embedding_store)
        marker_size = point_size if point_size is not None else 1
        marker_opacity = point_opacity if point_opacity is not None else 1.0

        figure = config_scatter_graph(
            labels[label_key],
            embeddings_dict[emb_name],
            marker_size=marker_size,
            marker_opacity=marker_opacity,
        )

        dataset_name = infoc_store.get('data_name', 'dataset') if infoc_store else 'dataset'
        title_text = f'{dataset_name} embedding'
        tooltip_text = f'Embedding computed on {emb_name}, cluster labels are computed/given by {label_key}'
        title_children = [
            html.Span(
                id='scatter-plot-title',
                children=title_text,
                style={'fontSize': '2rem', 'fontWeight': '600', 'color': '#555'},
            ),
            dbc.Tooltip(
                id='scatter-plot-tooltip',
                target='scatter-plot-title',
                placement='top',
                children=html.Span(tooltip_text, style={'fontSize': '1.5rem'}),
            ),
        ]

        return figure, title_children, html.Span(tooltip_text, style={'fontSize': '1.5rem'})

    _default_kde_title, _default_kde_body = config_selected_explanations()

    @app.callback(
        [Output('selected-explanation-title', 'children'), Output('selected-explanation', 'children')],
        Input('embedding-scatterPlot', 'selectedData'),
        [
            State('infoclus_store', 'data'),
            State('dataset_store', 'data'),
        ]
    )
    def select_embedding(selected, infoc_dict, data_store):

        if selected is None:
            return _default_kde_title, _default_kde_body
        if len(selected["points"]) == 0:
            return _default_kde_title, _default_kde_body

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

        title_children, body_children = config_selected_explanations(infoc_para_res=infoc_dict, df_data=deserialize_obj(data_store), selected_idxes=selected_idxes,ics_cluster=ic_selected, attributes=attributes_total[0])
        return title_children, body_children

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

    # LLM Modal callbacks
    @app.callback(
        Output('llm-modal', 'is_open'),
        Input('llm-button', 'n_clicks'),
        State('llm-modal', 'is_open'),
        prevent_initial_call=True,
    )
    def toggle_llm_modal(llm_btn_clicks, is_open):
        if llm_btn_clicks is None:
            raise dash.exceptions.PreventUpdate
        return not is_open

    @app.callback(
        Output('llm-prompt-output', 'children'),
        Output('llm-query-btn', 'style'),
        Input('llm-create-prompt-btn', 'n_clicks'),
        [State('infoclus_store', 'data'),
         State('dataset_store', 'data'),
         State('llm-task-input', 'value')],
        prevent_initial_call=True,
    )
    def generate_prompt(n_clicks, infoclus_store, dataset_store, task):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        prompt_text = create_prompt(
            infoclus_store=infoclus_store,
            dataset_store=dataset_store,
            task=task,
        )
        return prompt_text, {'display': 'inline-block', 'marginTop': '0.75rem'}

    @app.callback(
        Output('llm-query-output', 'children'),
        Input('llm-query-btn', 'n_clicks'),
        [State('llm-provider-input', 'value'),
         State('llm-api-key-input', 'value'),
         State('llm-model-input', 'value'),
         State('llm-endpoint-input', 'value'),
         State('llm-prompt-output', 'children')],
        prevent_initial_call=True,
    )
    def run_llm_query(n_clicks, provider, api_key, model, endpoint, prompt_text):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        return query_llm(
            provider=provider,
            api_key=api_key,
            model=model,
            prompt=prompt_text,
            endpoint=endpoint,
        )

    @app.callback(
        Output('clustering_store', 'data', allow_duplicate=True),
        Output('clustering-dropdown-menu', 'children', allow_duplicate=True),
        Output('clustering-value-store', 'data', allow_duplicate=True),
        Output('llm-query-output', 'children', allow_duplicate=True),
        Input('llm-update-labels-btn', 'n_clicks'),
        [State('llm-query-output', 'children'),
         State('clustering_store', 'data'),
         State('infoclus_store', 'data')],
        prevent_initial_call=True,
    )
    def apply_llm_labels(n_clicks, llm_response, labels, infoclus_store):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        if not isinstance(labels, dict):
            return dash.no_update, dash.no_update, dash.no_update, "Cannot update labels: clustering store is invalid."

        base_infoclus_labels = labels.get('InfoClus')
        if base_infoclus_labels is None and isinstance(infoclus_store, dict):
            base_infoclus_labels = infoclus_store.get('clustering')

        if base_infoclus_labels is None:
            return dash.no_update, dash.no_update, dash.no_update, "Cannot update labels: InfoClus labels are missing."

        try:
            unique_cluster_ids = sorted({int(v) for v in base_infoclus_labels})
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update, "Cannot update labels: InfoClus labels are not numeric cluster ids."

        mapping, err = parse_llm_cluster_labels(llm_response, expected_cluster_ids=unique_cluster_ids)
        if err is not None:
            return dash.no_update, dash.no_update, dash.no_update, f"Cannot update labels: {err}"
        if mapping is None:
            return dash.no_update, dash.no_update, dash.no_update, "Cannot update labels: failed to parse cluster labels."

        transformed_labels = [mapping[int(v)] for v in base_infoclus_labels]
        labels_update = dict(labels)
        labels_update['infoclus-labeled by LLM'] = transformed_labels

        menu_children_update = [
            dbc.DropdownMenuItem(k, id={'type': 'cluster-option', 'index': k})
            for k in labels_update.keys()
        ] + [
            dbc.DropdownMenuItem(
                html.Div(
                    dcc.Upload(
                        id='import-labels',
                        children=['Upload labels...'],
                        style={'cursor': 'pointer', 'padding': '0.25rem 0.5rem'},
                    ),
                    style={'width': '100%'},
                ),
                id='cluster-option-upload',
            )
        ]

        return labels_update, menu_children_update, 'infoclus-labeled by LLM', 'InfoClus labels updated from LLM response.'



    # @app.callback(
    #     Output('embedding-select', 'options'),
    #     Input('import-embedding', 'contents'),
    #     State('import-embedding', 'filename')
    # )
    # def import_embedding(contents, filename):
    #     if contents is None:
    #         return dash.no_update

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



