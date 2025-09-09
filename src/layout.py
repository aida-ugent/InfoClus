import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml, time
from sklearn.neighbors import KernelDensity
from dash import dcc, html
import dash_bootstrap_components as dbc

from dash_utils import *


RUNTIME_MARKERS = ["0.01s", "0.5s", "1s", "5s", "10s", "30s", "1m","3m", "5m", "10m", "30m", "1h"]

SIDEBAR_STYLE = {
    "overflow-y": "scroll",
    "height": "800px"
}

top_bar_style={'display': 'inline-block', 'padding': '5px 10px',
                                                   'background-color': '#e0e0e0', 'border-radius': '5px',
                                                   'font-size': '14px', 'margin-right': '10px'}

KERNALS = ["scott", "silverman"]
KERNAL = KERNALS[0]

INFOCLUS_OBJ = None

def get_kde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str, ic):
    """
    :return: return kernal desity estimation of one attribute for a cluster
    """
    percentage = len(cluster_att) / len(data_att)
    # Note: two kde's need to have the same bandwidth to ensure that they are comparable
    kde_data = KernelDensity(kernel='gaussian', bandwidth=KERNAL).fit(data_att.reshape(-1,1))
    kde_cluster = KernelDensity(kernel='gaussian', bandwidth=kde_data.bandwidth_).fit(cluster_att.reshape(-1,1))

    x_vals = np.linspace(min(min(data_att), min(cluster_att)), max(max(data_att), max(cluster_att)), 1000)
    kde_data_vals = np.exp(kde_data.score_samples(x_vals.reshape(-1, 1)))
    kde_cluster_vals = np.exp(kde_cluster.score_samples(x_vals.reshape(-1, 1)))

    cluster_proportion = len(cluster_att) / len(data_att)
    overlap_density = kde_cluster_vals * cluster_proportion

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=kde_data_vals, mode='lines', name=f'kde of {att_name} on full data',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=x_vals, y=kde_cluster_vals, mode='lines', name=f'kde of {att_name} on cluster',
                             line=dict(color='green', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=x_vals, y=overlap_density, fill='tozeroy', name=f'{percentage}% Overlapped by Cluster',
                             line=dict(color='orange', width=1)))

    fig.update_layout(
        xaxis=dict(
            title=att_name + "-IC-"+str(round(ic,1)),
            showline=True,
            linecolor="gray",
            linewidth=1
        ),
        yaxis=dict(
            # title="Densities",
            showline=True,
            linecolor="gray",
            linewidth=1
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

def config_scatter_graph(clustering: list, embedding: np.ndarray):

    # clustering = infoc_para_res['clustering']

    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': pd.Categorical(clustering),  # Classifications
        'customdata': list(range(len(embedding))),
    })

    fig = px.scatter(df, x='x', y='y', color='class', custom_data=['customdata'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

def config_explanations(infoc_para_res: dict, df_data: pd.DataFrame, cluster_label: int = 0):
    """
    :return: kde distributions for all selected features in a cluster, default as 0
    """
    scaled_data = np.array(infoc_para_res['scaled_data'])
    instance_cluster_idx = infoc_para_res['clusters_idxes_opt'][cluster_label]
    cluster = scaled_data[instance_cluster_idx]
    percentage = len(instance_cluster_idx)/df_data.shape[0] * 100

    figures = []
    figures.append(html.Br())
    figures.append(dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info"))

    att_names = df_data.columns
    ics_cluster = np.array(infoc_para_res['ic_opt'][cluster_label])
    for att_id in infoc_para_res['attributes_opt'][cluster_label]:
        data_att = scaled_data[:, att_id]
        cluster_att = cluster[:, att_id]
        att_name = att_names[att_id]
        if infoc_para_res['global_arr_type'] == 'categorical':
            # fig = get_barchart(infoclus, att_id, cluster_label, att_name)
            pass
        elif infoc_para_res['global_arr_type'] == 'numeric':
            fig = get_kde(data_att, cluster_att, att_name, ics_cluster[att_id])
        else:
            print('unsupported attribute type for visualization:', infoc_para_res['global_arr_type'])

        # figures.append(html.H6([att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id=f"Cluster {cluster_label}, {att_name}",
                                 figure=fig,
                                 style = {'width': '100%', 'height': '40%'},
                                 config = {'responsive': True}
                                 )
                       )

    return figures

def config_selected_explanations(infoc_para_res: dict = None, df_data: pd.DataFrame=None, selected_idxes=None, ics_cluster=None, attributes=None):

    if selected_idxes is None:
        return 'exploring dataset by selecting points by lasso in the above scatter plot '

    scaled_data = np.array(infoc_para_res['scaled_data'])
    cluster = scaled_data[selected_idxes]
    percentage = len(selected_idxes) / df_data.shape[0] * 100

    figures = []
    figures.append(html.Br())
    figures.append(dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info"))

    att_names = df_data.columns

    for att_id in attributes:
        data_att = scaled_data[:, att_id]
        cluster_att = cluster[:, att_id]
        att_name = att_names[att_id]
        if infoc_para_res['global_arr_type'] == 'categorical':
            # fig = get_barchart(infoclus, att_id, cluster_label, att_name)
            pass
        elif infoc_para_res['global_arr_type'] == 'numeric':
            fig = get_kde(data_att, cluster_att, att_name, ics_cluster[att_id])
        else:
            print('unsupported attribute type for visualization:', infoc_para_res['global_arr_type'])
        # figures.append(html.H6(
        #     [att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(
                                 figure=fig,
                                 style={'height': '100%', 'aspect-ratio': '1.3'},
                                 config={'responsive': True}
                                 )
                       )

    return figures

def get_embedding_dropdown_items():
    items =[
        {'label': 'tsne', 'value': 'tsne'},
        {'label': 'pca', 'value': 'pca'}
    ]
    return items

def get_runtime_dropdown_items():
    items =[
        {'label': 'recalculate in 1 s', 'value': '1'},
        {'label': 'recalculate in 10 s', 'value': '4'},
        {'label': 'recalculate in 30 s', 'value': '5'}
    ]
    return items

def get_clustering_dropdown_items(labels: dict):
    items=[]
    for key in labels.keys():
        items.append({'label': key, 'value': key})
    return items

def get_auxiliary_text_for_clustering():
    return "The clustering result is computed under parameters ..."

def config_layout(infoc_para_res: dict, df_data: pd.DataFrame, embeddings: dict, labels: dict, cluster_id: int = 0):

    dataset_name = infoc_para_res['data_name']

    count_clusters = infoc_para_res['count_clusters']
    main_emb_name = infoc_para_res['emb_name']

    return html.Div([

        dbc.Row(
            id='layout',
            children=[
                dbc.Col(
                    xs=12,
                    sm=3,
                    md=3,
                    id='selection-panel',
                    children=dbc.Card(
                        dbc.CardBody([
                        html.Div(
                            children=[
                                dbc.Row('Welcome to InfoClus, '
                                        'a new clustering method that also explains its clusters. '
                                        'Play with existed datasets or import your own dataset!',
                                        id='welcome-block'),
                                html.Br(),

                                dbc.Card(
                                    dbc.CardBody(children=[
                                        dcc.Upload(
                                            id='import-dataset',
                                            children=html.Button('Upload Dataset'),
                                        ),
                                        dcc.Upload(
                                            id='import-embedding',
                                            children=html.Button('Upload Embedding')
                                        ),
                                    ])
                                ),
                                html.Br(),

                                dbc.Card(
                                    dbc.CardBody(children=[
                                        dbc.Row(
                                            children=[
                                                html.Span(
                                                    children=[
                                                        'Select dataset: ',
                                                        dcc.Dropdown(
                                                            options=[{'label': dataset, 'value': dataset} for dataset in get_datasets()],
                                                            value=dataset_name,
                                                            id='dataset-select',
                                                            style={'width': '15em',
                                                                   'display': 'inline-block',
                                                                   'verticalAlign': 'middle'
                                                                   }
                                                        )
                                                    ],
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            children=[
                                                html.Span(
                                                    children=[
                                                        'Select embedding: ',
                                                        dcc.Dropdown(
                                                            options=get_embedding_dropdown_items(),
                                                            value=main_emb_name,
                                                            id='embedding-select',
                                                            style={'width': '13em',
                                                                   'display': 'inline-block',
                                                                   'verticalAlign': 'middle'},
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ])
                                ),
                                html.Br(),

                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H5('Hyper-parameters tuning', className='card-title'),
                                            dbc.Row(
                                                [
                                                    dbc.Col('alpha', width='auto'),
                                                    dbc.Col(children=dcc.Slider(
                                                                        id='alpha-slider',
                                                                        min=int(infoc_para_res['alpha']/5),
                                                                        max=int(infoc_para_res['alpha']*5),
                                                                        marks={
                                                                            int(infoc_para_res['alpha'] / 5): {'label': str(int(infoc_para_res['alpha'] / 5))},
                                                                            int(infoc_para_res['alpha'] * 5): {'label': str(int(infoc_para_res['alpha'] * 5))},
                                                                        },
                                                                        step=1,
                                                                        value=infoc_para_res['alpha'],
                                                                        tooltip={"always_visible": True, 'placement': 'bottom'},
                                                                    ),
                                                            ),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col('beta', width='auto'),
                                                    dbc.Col(children=dcc.Slider(
                                                                        id='beta-slider',
                                                                        min=1,
                                                                        max=2,
                                                                        step=0.1,
                                                                        marks={
                                                                            1: {'label': str(1)},
                                                                            2: {'label': str(2)},
                                                                        },
                                                                        value=infoc_para_res['beta'],
                                                                        tooltip={"always_visible": True, 'placement': 'bottom'}
                                                                    ),
                                                            ),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col('min_att', width='auto'),
                                                    dbc.Col(children=dcc.Input(type="number", value=2, step=1, min=0,max=10, id='min-att-input'),
                                                            width='auto'),
                                                    dbc.Col('max_att', width='auto'),
                                                    dbc.Col(children=dcc.Input(type="number", value=5, step=1, min=3, max=10, id='max-att-input',),
                                                            width='auto'),
                                                ],
                                            ),
                                            dcc.Dropdown(
                                                value ='1',
                                                options=get_runtime_dropdown_items(),
                                                id = 'recalc-hyperparameters'
                                            )
                                        ]
                                    ),
                                    color='white'
                                )])]),
                        style={
                            'height': '90vh',
                            'overflowY': 'auto',
                        })
                ),
                dbc.Col(
                    xs=12,
                    sm=6,
                    md=6,
                    id = 'clustering-panel',
                    children=dbc.Card(
                        dbc.CardBody([
                        html.Div(
                            [
                                dbc.Row(
                                children=[
                                    html.Span(children=[

                                        dcc.Dropdown(
                                            options=get_clustering_dropdown_items(labels),
                                            value = 'infoclus_clustering',
                                            id = 'clustering-to-show-select',
                                            style={'width': '8em',
                                                   'display': 'inline-block',
                                                   'verticalAlign': 'middle'
                                                   }
                                        ),

                                        ' shown on embedding ',

                                         dcc.Dropdown(
                                             options=get_embedding_dropdown_items(),
                                             value=main_emb_name,
                                             id='embedding-for-show',
                                             style={'width': '8em',
                                                    'display': 'inline-block',
                                                    'verticalAlign': 'middle'
                                                    }
                                         ),

                                        dcc.Upload(
                                            id='import-labels',
                                            children=html.Button('Upload labels'),
                                        )

                                    ], ),
                                ],
                                ),
                                dbc.Tooltip(
                                    get_auxiliary_text_for_clustering(),
                                    target='clustering-text',
                                    placement='top'
                                ),
                                dcc.Graph(
                                    id="embedding-scatterPlot",
                                    figure=config_scatter_graph(labels['infoclus_clustering'], embeddings[main_emb_name]),
                                    style={ 'height': '50vh'},
                                    # config={"editable": False, "modeBarButtonsToAdd": ["lasso2d", "select2d"]},
                                ),
                                # dcc.Markdown(
                                #     r"$R_{\alpha,\beta}(\mathcal{C}, \mathcal{E}) = \frac{\sum_{i=1}^r{\sum_{j=1}^{|e_i|}{I_i^j}}}{\alpha + (\sum_{i=1}^r{\sum_{j=1}^{|e_i|}{|a_i^j|}})^\beta}$ is ...",
                                #     mathjax=True
                                # ),
                                dbc.Row(
                                    # dbc.Alert("Contains " + format(len(selected_idxes) / df_data.shape[0] * 100, '.2f') + ' % of data', color="info")
                                    dbc.Col(
                                        id = 'selected-explanation',
                                        children=config_selected_explanations(),
                                        style={
                                            'display': 'flex',
                                            'height': '30vh',
                                            'autoflowX': 'auto',
                                        })
                                )
                            ],)]),
                        style={
                            'height': '90vh',
                            'overflowY': 'auto'
                        })
                ),
                dbc.Col(
                    xs=12,
                    sm=3,
                    md=3,
                    id='explanation-panel',
                    children=dbc.Card(
                        dbc.CardBody([
                        html.Div(
                            [html.Span(
                            [html.H5("Cluster explanation"),
                             dcc.Dropdown(
                                 id='cluster-select',
                                 options=[
                                     {'label': "Cluster " + str(i), 'value': i} for i in range(infoc_para_res['count_clusters'])
                                 ],
                                 value=cluster_id
                             ),]),
                        dbc.Row(id='explanation',
                                children=config_explanations(infoc_para_res, df_data, cluster_id),
                                style={
                                    'height': '70vh',
                                    'overflowY': 'auto'
                                }
                                )]


                        )

                    ]),
                        style={
                            'height': '90vh',
                            'overflowY': 'auto'
                        }
                    )
                )
            ]
        )
        ])

#
# def get_barchart(infoclus: InfoClus, att_id: int, cluster_id: int, att_name: str):
#
#     df_mapping_chain = infoclus.ls_mapping_chain_by_col[att_id]
#     real_labels = df_mapping_chain.iloc[:,0]
#     nuniques = len(df_mapping_chain)
#     dist_of_fixed_cluster_att = infoclus._clustersRelatedInfo[cluster_id][0].iloc[:nuniques, att_id].values
#     dist_of_att_in_data = infoclus._priors.iloc[:nuniques, att_id].values
#
#     dist_pre_cluster_att = pd.Series(dist_of_fixed_cluster_att, index=real_labels)
#     dist_prior_per_att = pd.Series(dist_of_att_in_data, index=real_labels)
#     sorted_dist_pre_cluster_att = dist_pre_cluster_att.sort_values(ascending=False)
#     sorted_dist_prior_per_att = dist_prior_per_att.loc[sorted_dist_pre_cluster_att.index]
#     sorted_labels = sorted_dist_pre_cluster_att.index
#     sorted_distribution = []
#     types = []
#     group_labels = []
#     for label in sorted_labels:
#         sorted_distribution.append(sorted_dist_pre_cluster_att[label])
#         sorted_distribution.append(sorted_dist_prior_per_att[label])
#         types.extend(['Cluster', 'Prior'])
#         group_labels.extend([label, label])
#
#     data = pd.DataFrame({
#         "Labels": group_labels,
#         "Distribution": sorted_distribution,
#         "Type": types
#     })
#     fig = px.bar(
#         data,
#         x="Labels",
#         y="Distribution",
#         color="Type",
#         barmode="group",
#         # title=f"Cluster {cluster_id} - Attribute {att_id}",
#         labels={"Distribution": "Distribution", "Labels": "Labels"}
#     )
#     fig.update_layout(
#         width=600,
#         height=400
#     )
#
#     return fig
