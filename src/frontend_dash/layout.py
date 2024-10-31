import plotly.express as px
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yaml
import anndata as ad
from scipy.stats import gaussian_kde

from dash import dcc
from dash import html
from ..caching import from_cache

RUNTIME_MARKERS = ["0.01s", "0.5s", "1s", "5s", "10s", "30s", "1m","3m", "5m", "10m", "30m", "1h", "full"]

SIDEBAR_STYLE = {
    "overflow-y": "scroll",
    "height": "680px"
}


def get_kde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str):
    """
    :return: return kernal desity estimation of one attribute for a cluster
    """

    kde_data = gaussian_kde(data_att)
    kde_cluster = gaussian_kde(cluster_att)

    x_vals = np.linspace(min(data_att), max(data_att), 1000)
    kde_data_vals = kde_data(x_vals)
    kde_cluster_vals = kde_cluster(x_vals)

    cluster_proportion = len(cluster_att) / len(data_att)
    overlap_density = kde_cluster_vals * cluster_proportion

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=kde_data_vals, mode='lines', name='Full Data Density',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=x_vals, y=kde_cluster_vals, mode='lines', name='Cluster Density',
                             line=dict(color='green', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=x_vals, y=overlap_density, fill='tozeroy', name='Overlap by Cluster',
                             line=dict(color='orange', width=2)))
    fig.update_layout(title=f"Density Distributions of {att_name}",
                      xaxis_title="Value",
                      yaxis_title="Densities",
                      showlegend=True)

    return fig


def config_scatter_graph(clustering: np.ndarray, embedding: np.ndarray):

    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': clustering  # Classifications
    })

    graph = dcc.Graph(
        id="embedding-scatterPlot",
        figure=px.scatter(df, x='x', y='y', color='class', title='clustering')
    )

    return graph


def config_explanations_kde(data: np.ndarray,
                            clustering: np.ndarray, attributes: list,
                            att_names: np.ndarray, ics_cluster: np.ndarray, cluster_label: int = 0):
    """
    :return: kde distributions for all selected features in a cluster, default as 0
    """

    instance_cluster_idx = np.where(clustering == cluster_label)
    cluster = data[instance_cluster_idx]
    percentage = instance_cluster_idx.shape[0]/data.shape[0]

    figures = [html.Br(), dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info")]

    for att in attributes:
        data_att = data[:, att]
        cluster_att = cluster[:, att]
        att_name = att_names[att]
        kde = get_kde(data_att, cluster_att, att_name)

        figures.append(html.H6(
            [att_name, dbc.Badge(format(ics_cluster[0, att], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id=f"Cluster {cluster_label}, {att_name}",
                                 figure=kde,
                                 config={
                                     'displayModeBar': False
                                 }))

    return figures


def config_hyperparameter_tuning():
    return dbc.Row(
        [
            dbc.Col(
                [
                    # Alpha
                    dbc.Row(
                        [
                            dbc.Col(html.H6(u"\u03B1"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='alpha-slider',
                                    min=0,
                                    max=500,
                                    step=10,
                                    marks={i: str(i) for i in range(0, 501, 50)},
                                    value=250,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                    # Beta
                    dbc.Row(
                        [
                            dbc.Col(html.H6(u"\u03B2"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='beta-slider',
                                    min=1.0,
                                    max=2.0,
                                    step=0.05,
                                    marks={round(i, 1): format(i, '.1f') for i in
                                           np.arange(1.0, 2.1, 0.1)},
                                    value=1.6,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                    # Runtime
                    dbc.Row(
                        [
                            dbc.Col(html.H6("runtime"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='runtime-slider',
                                    min=0,
                                    max=len(RUNTIME_MARKERS) - 1,
                                    step=1,
                                    marks={i: RUNTIME_MARKERS[i] for i in range(len(RUNTIME_MARKERS))},
                                    value=0,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                    # Min Attributes
                    dbc.Row(
                        [
                            dbc.Col(html.H6("minAtt"), width=2),
                            dbc.Col(
                                dcc.Slider(
                                    id='minAtt-slider',
                                    min=1,
                                    max=5,
                                    step=1,
                                    marks={i: str(i) for i in range(1, 5, 1)},
                                    value=2,
                                    tooltip={"always_visible": False}
                                )
                            )
                        ]
                    ),
                ],
                align="center",
            ),
            dbc.Col(
                [
                    dbc.Row(dbc.Col(
                        # Recalc restarts from scratch
                        dbc.Button("Recalc", color="primary", size="md", id="recalc-hyperparameters", block=True)),
                        justify="center"
                    ),
                    dbc.Tooltip(
                        "Restart calculation from scratch",
                        target="recalc-hyperparameters",
                        placement="right"
                    )
                ],
                align="center", width="auto"
            )
        ],
        justify="center"
    )


def get_dataset_path(data, dataset_name):
    # Find the dataset with the specified name and return its path in yaml
    for dataset in data['datasets']:
        if dataset['name'] == dataset_name:
            return dataset['path']
    return None


def get_dataset_main_emb(data, dataset_name):
    # Find the dataset with the specified name and return its path in yaml
    for dataset in data['datasets']:
        if dataset['name'] == dataset_name:
            return dataset['main_emb']
    return None


def config_layout(datasets_config: str = 'datasets_info.yaml', dataset_name: str = 'german_socio_eco', cluster_id: int = 0):

    with open(datasets_config, 'r') as file:
        datasets_info = yaml.safe_load(file)

    dataset_path = get_dataset_path(datasets_info, dataset_name)
    adata = ad.read_h5ad(dataset_path)

    count_clusters = len(np.unique(adata.obs['infoclus_clustering'].values))
    main_emb_name = adata.uns['InfoClus']['main_emb']
    clustering = adata.obs['infoclus_clustering'].values
    main_emb = adata.obsm.get(main_emb_name)
    data = adata.X
    att_names = adata.var.index.values
    attributes = adata.uns['InfoClus'][f'cluster_{cluster_id}']['attributes']
    ics_cluster = adata.uns['InfoClus'][f'cluster_{cluster_id}']['ic']

    return html.Div([
        # Top Navbar
        dbc.Row(dbc.Col(dbc.NavbarSimple(brand="InfoClus", color="primary", dark=True, fluid=True, sticky="top"))),
        # Dashboard with general info
        dbc.Card(
            dbc.CardBody(
                [
                    # Dashboard
                    dbc.Row(
                        dbc.Col(
                            dcc.Dropdown(
                                id='dataset-select',
                                options=[
                                    {'Dataset': dataset['name'], 'value': dataset['name']} for dataset in datasets_info['datasets']
                                ],
                                value=datasets_info['datasets'][0]['name']
                            )
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='embedding-select',
                                options=[
                                    {'Embedding': embedding['method'], 'value': embedding['method']} for embedding in datasets_info['embeddings']
                                ],
                                value=main_emb_name
                            )
                        ),
                        justify="center"
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            # Scatter plot and hyperparameter tuning
                            dbc.Col(
                                [
                                    # Scatter plot
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(children=dataset_name, className="card-title"),
                                                config_scatter_graph(clustering, main_emb)
                                            ]
                                        )
                                    ),
                                    html.Br(),
                                    # Hyperparameter tuning
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(children="Tune hyperparameters", className="card-title"),
                                                config_hyperparameter_tuning()
                                            ]
                                        )
                                    ),
                                ],
                                width="auto"
                            ),

                            # Explanation
                            dbc.Col(
                                [
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(children="Cluster explanation", className="card-title"),
                                                dcc.Dropdown(
                                                    id='cluster-select',
                                                    options=[
                                                        {'label': "Cluster " + str(i), 'value': i} for i in range(count_clusters)
                                                    ],
                                                    value=0
                                                ),
                                                html.Div(config_explanations_kde(data, clustering, attributes, att_names, ics_cluster, cluster_id), id="explanation", style=SIDEBAR_STYLE)
                                            ]
                                        ),
                                    )
                                ],
                                width="auto"
                            )
                        ], align="center", justify="center"
                    )
                ]
            )
        )
    ])


