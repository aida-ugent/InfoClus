import pickle
import subprocess
import plotly.express as px
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yaml
import sys, os

# def get_root():
#     try:
#         root = subprocess.check_output(
#             ["git", "rev-parse", "--show-toplevel"],
#             universal_newlines=True
#         ).strip()
#         return root
#     except subprocess.CalledProcessError:
#         raise RuntimeError("This directory is not a Git repository.")

def get_root():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up the directory tree until you find a specific file (like `setup.py` or a config file)
    # In this case, we can stop when we find the root of the project (you can customize the condition)
    while not os.path.exists(os.path.join(current_dir,
                                          'readme.md')):  # You can change 'setup.py' to something else (e.g., README.md or a custom marker file)
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Stop when you reach the root of the filesystem
            raise RuntimeError("Project root not found.")
        current_dir = parent_dir

    return current_dir
ROOT_DIR = get_root()
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(ROOT_DIR, "src", "frontend_dash"))

from dash import dcc
from dash import html
from src.infoclus import InfoClus
from sklearn.neighbors import KernelDensity


RUNTIME_MARKERS = ["0.01s", "0.5s", "1s", "5s", "10s", "30s", "1m","3m", "5m", "10m", "30m", "1h"]

SIDEBAR_STYLE = {
    "overflow-y": "scroll",
    "height": "800px"
    # "width": "fit-content"
}

top_bar_style={'display': 'inline-block', 'padding': '5px 10px',
                                                   'background-color': '#e0e0e0', 'border-radius': '5px',
                                                   'font-size': '14px', 'margin-right': '10px'}

KERNALS = ["gaussian", "tophat", "epanechnikov"]
KERNAL = KERNALS[0]

def get_kde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str):
    """
    :return: return kernal desity estimation of one attribute for a cluster
    """
    percentage = len(cluster_att) / len(data_att)
    # Note: two kde's need to have the same bandwidth to ensure that they are comparable
    kde_data = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_att.reshape(-1,1))
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
    fig.update_layout(xaxis_title="Value",
                      yaxis_title="Densities",
                      showlegend=True,
                      width=600,  # Set the figure width in pixels
                      height=400
                      )

    return fig

def get_barchart(infoclus: InfoClus, att_id: int, cluster_id: int, att_name: str):

    df_mapping_chain = infoclus.ls_mapping_chain_by_col[att_id]
    real_labels = df_mapping_chain.iloc[:,0]
    nuniques = len(df_mapping_chain)
    dist_of_fixed_cluster_att = infoclus._clustersRelatedInfo[cluster_id][0].iloc[:nuniques, att_id].values
    dist_of_att_in_data = infoclus._priors.iloc[:nuniques, att_id].values

    dist_pre_cluster_att = pd.Series(dist_of_fixed_cluster_att, index=real_labels)
    dist_prior_per_att = pd.Series(dist_of_att_in_data, index=real_labels)
    sorted_dist_pre_cluster_att = dist_pre_cluster_att.sort_values(ascending=False)
    sorted_dist_prior_per_att = dist_prior_per_att.loc[sorted_dist_pre_cluster_att.index]
    sorted_labels = sorted_dist_pre_cluster_att.index
    sorted_distribution = []
    types = []
    group_labels = []
    for label in sorted_labels:
        sorted_distribution.append(sorted_dist_pre_cluster_att[label])
        sorted_distribution.append(sorted_dist_prior_per_att[label])
        types.extend(['Cluster', 'Prior'])
        group_labels.extend([label, label])

    data = pd.DataFrame({
        "Labels": group_labels,
        "Distribution": sorted_distribution,
        "Type": types
    })
    fig = px.bar(
        data,
        x="Labels",
        y="Distribution",
        color="Type",
        barmode="group",
        # title=f"Cluster {cluster_id} - Attribute {att_id}",
        labels={"Distribution": "Distribution", "Labels": "Labels"}
    )
    fig.update_layout(
        width=600,
        height=400
    )

    return fig

def config_scatter_graph(infoclus: InfoClus, emb_name: str = None):
    if emb_name is None:
        emb_name = infoclus.emb_name

    alpha = infoclus.alpha
    beta = infoclus.beta
    mina = infoclus.min_att
    maxa = infoclus.max_att
    runid = infoclus.runtime_id
    run_time_marker = RUNTIME_MARKERS[runid]

    clustering = infoclus._clustering_opt
    embedding = infoclus.all_embeddings[emb_name]

    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': pd.Categorical(clustering)  # Classifications
    })
    #todo: error, change the title to another bar, update with optimise and dataset switch
    fig = px.scatter(df, x='x', y='y', color='class')
    fig.update_layout(
        width=810,
        height=540
    )
    return fig


def config_explanations(infoclus: InfoClus, data: np.ndarray,
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

    for att_id in attributes:
        data_att = data[:, att_id]
        cluster_att = cluster[:, att_id]
        att_name = att_names[att_id]
        att_type = infoclus.var_type[att_id]
        if att_type == 'categorical':
            fig = get_barchart(infoclus, att_id, cluster_label, att_name)
        elif att_type == 'numeric':
            fig = get_kde(data_att, cluster_att, att_name)
        else:
            print('unsupported attribute type for visualization:', att_type)

        figures.append(html.H6(
            [att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id=f"Cluster {cluster_label}, {att_name}",
                                 figure=fig,
                                 config={
                                     'displayModeBar': False
                                 }))

    return figures


def config_hyperparameter_tuning(infoclus: InfoClus):

    alpha = infoclus.alpha
    alpha_max = alpha * 5
    alpha_min = 0

    beta = infoclus.beta
    beta_max = 2
    beta_min = 1

    mina = infoclus.min_att
    mina_max = mina * 5
    mina_min = 0
    maxa = infoclus.max_att
    maxa_max = maxa * 5
    maxa_min = 0

    runid = infoclus.runtime_id
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


def config_layout(datasets_config: str = 'datasets_info.yaml', dataset_name: str = 'german_socio_eco', emb_name: str = 'tsne', cluster_id: int = 0):

    with open(datasets_config, 'r') as file:
        datasets_info = yaml.safe_load(file)

    if dataset_name not in datasets_info['datasets']:
        print("Error! Dataset not found.")

    # read pickle from path
    data_path = os.path.join(ROOT_DIR, 'data', dataset_name, f'{dataset_name}_{emb_name}.pkl')
    if os.path.exists(data_path):
        with open(data_path, 'rb') as file:
            infoclus = pickle.load(file)
    else:
        infoclus = InfoClus(dataset_name=dataset_name, main_emb=emb_name)
        infoclus.optimise()

    clustering = infoclus._clustering_opt
    count_clusters = len(np.unique(infoclus._clustering_opt))
    main_emb_name = infoclus.emb_name
    main_emb = infoclus.embedding
    data = infoclus.data
    count_intances = data.shape[0]
    att_names = infoclus.data_raw.columns.values
    attributes = infoclus._attributes_opt[cluster_id]
    ics_cluster = np.array(infoclus._ic_opt[cluster_id])

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
                    dbc.Row(
                        html.Div([html.Span("embedding used for clustering: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='embedding-used-for-clustering', type='text',
                                            value=f"{infoclus.emb_name}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("alpha: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='alpha-value', type='text', value=f"{infoclus.alpha}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("beta: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='beta-value', type='text', value=f"{infoclus.beta}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("min_att: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='min-att', type='text', value=f"{infoclus.min_att}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("max_att: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='max-att', type='text', value=f"{infoclus.max_att}", readOnly=True,
                                            style=top_bar_style),
                                  html.Span("run time: ", style={'margin-right': '10px'}),
                                  dcc.Input(id='run time id', type='text', value=f"{RUNTIME_MARKERS[infoclus.runtime_id]}", readOnly=True,
                                            style=top_bar_style),
                                  ]),
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
                                                dcc.Graph(
                                                        id="embedding-scatterPlot",
                                                        figure=config_scatter_graph(infoclus, main_emb_name)
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
                                                config_hyperparameter_tuning(infoclus)
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
                                                html.Div(config_explanations(infoclus, data, clustering, attributes, att_names, ics_cluster, cluster_id),
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


