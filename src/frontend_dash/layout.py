from calendar import error

import plotly.express as px
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yaml
import anndata as ad
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import sys, os

from dash import dcc
from dash import html
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.caching import from_cache
from src.utils import get_git_root

ROOT_DIR = get_git_root()
RUNTIME_MARKERS = ["0.01s", "0.5s", "1s", "5s", "10s", "30s", "1m","3m", "5m", "10m", "30m", "1h"]

SIDEBAR_STYLE = {
    "overflow-y": "scroll",
    "height": "800px"
    # "width": "fit-content"
}

KERNALS = ["gaussian", "tophat", "epanechnikov"]
KERNAL = KERNALS[0]

def get_kde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str):
    """
    :return: return kernal desity estimation of one attribute for a cluster
    """

    # Note: two kde's need to have the same bandwidth to ensure that they are comparable
    kde_data = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_att.reshape(-1,1))
    kde_cluster = KernelDensity(kernel='gaussian', bandwidth=kde_data.bandwidth_).fit(cluster_att.reshape(-1,1))

    x_vals = np.linspace(min(min(data_att), min(cluster_att)), max(max(data_att), max(cluster_att)), 1000)
    kde_data_vals = np.exp(kde_data.score_samples(x_vals.reshape(-1, 1)))
    kde_cluster_vals = np.exp(kde_cluster.score_samples(x_vals.reshape(-1, 1)))

    cluster_proportion = len(cluster_att) / len(data_att)
    overlap_density = kde_cluster_vals * cluster_proportion

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=kde_data_vals, mode='lines', name='Full Data Density',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=x_vals, y=kde_cluster_vals, mode='lines', name='Cluster Density',
                             line=dict(color='green', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=x_vals, y=overlap_density, fill='tozeroy', name='Overlap by Cluster',
                             line=dict(color='orange', width=1)))
    fig.update_layout(xaxis_title="Value",
                      yaxis_title="Densities",
                      showlegend=True,
                      width=600,  # Set the figure width in pixels
                      height=400
                      )

    return fig


def config_scatter_graph(adata: ad.AnnData, emb_name: str = None):
    if emb_name is None:
        emb_name = adata.uns['InfoClus']['main_emb']
    alpha = adata.uns['InfoClus']['hyperparameters']['alpha']
    beta = adata.uns['InfoClus']['hyperparameters']['beta']
    mina = adata.uns['InfoClus']['hyperparameters']['mina']
    maxa = adata.uns['InfoClus']['hyperparameters']['maxa']
    runid = adata.uns['InfoClus']['hyperparameters']['runid']
    run_time_marker = RUNTIME_MARKERS[runid]

    clustering = adata.obs['infoclus_clustering'].values
    embedding = adata.obsm.get(emb_name)

    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': pd.Categorical(clustering)  # Classifications
    })
    fig = px.scatter(df, x='x', y='y', color='class',
                     title=f'clusteringEmb_{emb_name} alpha_{alpha} beta_{beta} mina_{mina} maxa_{maxa} runtime_{run_time_marker}')
    fig.update_layout(
        width=810,
        height=540
    )
    return fig


def config_explanations_kde(data: np.ndarray,
                            clustering: np.ndarray, attributes: list,
                            att_names: np.ndarray, ics_cluster: np.ndarray, cluster_label: int = 0):
    """
    :return: kde distributions for all selected features in a cluster, default as 0
    """
    # todo: optimise clustering as instance_cluster_idx in parameter transfer
    instance_cluster_idx = np.where(clustering == cluster_label)
    cluster = data[instance_cluster_idx]
    percentage = instance_cluster_idx[0].shape[0]/data.shape[0] * 100

    figures = [html.Br(), dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info")]

    for att in attributes:
        data_att = data[:, att]
        cluster_att = cluster[:, att]
        att_name = att_names[att]
        kde = get_kde(data_att, cluster_att, att_name)

        figures.append(html.H6(
            [att_name, dbc.Badge(format(ics_cluster[att], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id=f"Cluster {cluster_label}, {att_name}",
                                 figure=kde,
                                 config={
                                     'displayModeBar': False
                                 }))

    return figures


def config_hyperparameter_tuning(adata):

    alpha = adata.uns['InfoClus']['hyperparameters']['alpha']
    alpha_max = alpha * 5
    alpha_min = 0

    beta = adata.uns['InfoClus']['hyperparameters']['beta']
    beta_max = 2
    beta_min = 1

    mina = adata.uns['InfoClus']['hyperparameters']['mina']
    mina_max = mina * 5
    mina_min = 0
    maxa = adata.uns['InfoClus']['hyperparameters']['maxa']
    maxa_max = maxa * 5
    maxa_min = 0

    runid = adata.uns['InfoClus']['hyperparameters']['runid']
    runid_max = len(RUNTIME_MARKERS)
    runid_min = 0


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
                                    min=alpha_min,
                                    max=alpha_max,
                                    step=10,
                                    marks={i: str(i) for i in range(alpha_min, alpha_max, int((alpha_max-alpha_min)/10))},
                                    # todo: align initial values to be the same with initial webpage activation
                                    value=alpha,
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
                                    min=beta_min,
                                    max=beta_max,
                                    step=0.05,
                                    marks={round(i, 1): format(i, '.1f') for i in
                                           np.arange(beta_min, beta_max, 0.1)},
                                    value=beta,
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
                                    min=runid_min,
                                    max=runid_max,
                                    step=1,
                                    marks={i: RUNTIME_MARKERS[i] for i in range(runid_max)},
                                    value=runid,
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
                                    min=mina_min,
                                    max=mina_max,
                                    step=1,
                                    marks={i: str(i) for i in range(mina_max, 1)},
                                    value=mina,
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
                        dbc.Button("Recalc", color="primary", size="md", id="recalc-hyperparameters")),
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


def get_dataset_path(dataset_name):
    # Find the dataset with the specified name and return its path in yaml
    script_a_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_a_dir, '..', '..', 'data', dataset_name)
    if os.path.exists(path):
        return path
    else:
        print("Dataset not found")
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

    if dataset_name not in datasets_info['datasets']:
        print("Error! Dataset not found.")

    adata = ad.read_h5ad(os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}.h5ad'))

    count_clusters = len(np.unique(adata.obs['infoclus_clustering'].values))
    main_emb_name = adata.uns['InfoClus']['main_emb']
    clustering = adata.obs['infoclus_clustering'].values
    main_emb = adata.obsm.get(main_emb_name)
    data = adata.X
    count_intances = data.shape[0]
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
                    # Dropdowns
                    dbc.Row(
                        [
                        dbc.Col(
                            dcc.Dropdown(
                                id='dataset-select',
                                options=[
                                    {'label': dataset, 'value': dataset} for dataset in datasets_info['datasets']
                                ],
                                value=dataset_name
                            )
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='embedding-select',
                                options=[
                                    {'label': embedding, 'value': embedding} for embedding in datasets_info['embeddings']['method']
                                ],
                                value=main_emb_name
                            )
                        )
                    ]),
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
                                                dcc.Graph(
                                                        id="embedding-scatterPlot",
                                                        figure=config_scatter_graph(adata, main_emb_name)
                                                    )
                                            ]
                                        )
                                    ),
                                    html.Br(),
                                    # Hyperparameter tuning
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(children="Tune hyperparameters", className="card-title"),
                                                config_hyperparameter_tuning(adata)
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
                                                html.Div(config_explanations_kde(data, clustering, attributes, att_names, ics_cluster, cluster_id),
                                                         id="explanation", style=SIDEBAR_STYLE)
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


