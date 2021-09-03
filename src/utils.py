import numpy as np
import copy
import re
import pandas as pd
import scipy.stats as ss

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from caching import from_cache, to_cache

from si import ExclusOptimiser

SIDEBAR_STYLE = {
    "overflow-y": "scroll",
    "height": "680px"
}

MAIN_STYLE = {
    "height": "650px"
}

DASHBOARD_STYLE = {
    "text-align": "center",
}

RUNTIME_MARKERS = ["0.5s", "1s", "5s", "10s", "30s", "1m", "5m", "10m", "30m", "1h", "full"]


def compute_embedding(df_data, path_to_emb_file):
    data = from_cache(path_to_emb_file)
    if data is None:
        tsne = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000, learning_rate=200, random_state=2)
        embedding = tsne.fit_transform(df_data)
        data = {'Y': embedding}
        to_cache(path_to_emb_file, data)
    return data['Y']


def render_dashboard_card(title, new_value, old_value, old=False, color="primary"):
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H5(title, className="card-title")), justify="center"),
                dbc.Row([
                    dbc.Col([html.H6("New value"), dbc.Alert(children=new_value, color=color)]),
                    dbc.Col([html.H6("Old value"), dbc.Alert(children=old_value, color="secondary")])
                ])
            ],
        )
    )


def render_dashboard(cluster_ids, attributes, ic, cluster_ids_old=[], attributes_old=[], ic_old=0):
    old = False
    size = 0
    for row in attributes:
        size += len(row)
    size_old = 0
    for row in attributes_old:
        size_old += len(row)
    c_color = "primary"
    a_color = "primary"
    s_color = "primary"
    if cluster_ids_old:
        c_color = "success" if len(cluster_ids) >= len(cluster_ids_old) else "danger"
        a_color = "success" if size >= size_old else "danger"
        s_color = "success" if ic >= ic_old else "danger"
        old = True
    return dbc.CardDeck(
        [
            # Card showing # of clusters and change in #
            render_dashboard_card("# clusters", len(cluster_ids), len(cluster_ids_old), old=old, color=c_color),
            # Shows total # of attributes and change in #
            render_dashboard_card("# attributes", size, size_old, old=old, color=a_color),
            # Shows subjective interestingness and change
            render_dashboard_card("Information content", round(ic), round(ic_old), old=old, color=s_color),
        ], style=DASHBOARD_STYLE
    )


def config_figure(embedding, labels):
    tuples = list(zip(embedding[:, 0], embedding[:, 1], labels))
    df = pd.DataFrame(tuples, columns=['x', 'y', 'cluster'])
    df = df.sort_values('cluster')
    df["cluster"] = df["cluster"].astype(str)
    figure = px.scatter(df, x='x', y='y', color='cluster',
                        color_discrete_sequence=px.colors.qualitative.Dark24,
                        height=500)
    figure['layout'].update(autosize=False)
    figure.update_layout(margin=dict(l=0, r=150, b=0, t=0))
    figure.update_yaxes(automargin=True)
    figure.update_yaxes(visible=False, showticklabels=False, mirror=True, showline=True)
    figure.update_xaxes(visible=False, showticklabels=False, mirror=True, showline=True)
    return figure


def config_scatter(embedding, labels):
    return dcc.Graph(
        id='scatter',
        figure=config_figure(embedding, labels),
        config={'displayModeBar': False,
                'displaylogo': False}

    )


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

                ],
                align="center",
            ),
            dbc.Col(
                [
                    dbc.Row(dbc.Col(
                        # Refine starts from current clustering
                        dbc.Button("Refine", color="primary", size="md", id="refine-hyperparameters", block=True)),
                        justify="center"
                    ),
                    dbc.Tooltip(
                        "Refine calculation from current clustering",
                        target="refine-hyperparameters",
                        placement="right"
                    ),
                    html.Br(),
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


def config_explanation(df_data, labels, attributes, priors, dls, ics, cluster=0):
    cluster_data = df_data.iloc[np.nonzero(labels == cluster)[0], :]
    percentage = cluster_data.shape[0] / df_data.shape[0] * 100
    column_names = df_data.columns
    means = cluster_data.mean()
    stds = cluster_data.std()
    figures = [html.Br(), dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info")]
    for attribute in attributes[cluster]:
        # DL = 2 so show to normal distributed curves (prior and cluster)
        if dls[attribute] == 2:
            min_val = min(means.iloc[attribute] - 4 * stds.iloc[attribute],
                          priors[column_names[attribute]][0] - 4 * priors[column_names[attribute]][1])
            max_val = max(means.iloc[attribute] + 4 * stds.iloc[attribute],
                          priors[column_names[attribute]][0] + 4 * priors[column_names[attribute]][1])
            x = np.linspace(min_val, max_val, 1000)
            epsilon = 0
            if stds.iloc[attribute] == 0:
                epsilon = (max_val - min_val) / 100
            y_cluster = ss.norm.pdf(x, means.iloc[attribute], stds.iloc[attribute] + epsilon)
            y_prior = ss.norm.pdf(x, priors[column_names[attribute]][0], priors[column_names[attribute]][1])
            plot_data = {column_names[attribute]: np.concatenate((x, x)),
                         "pdf": np.concatenate((y_cluster, y_prior)),
                         'labels': ['cluster'] * 1000 + ['all data'] * 1000}
            df_explanation = pd.DataFrame(plot_data)
            name = column_names[attribute]
            fig = px.line(df_explanation, x=column_names[attribute], y="pdf", color='labels', width=400, height=300)
        # DL = 1 and it is a binary attribute, so show 2 stacked bar plots (cluster and prior)
        else:
            column_name = column_names[attribute]
            name = re.findall(r"^(.*?)(?=\s\()", column_name)[0]
            label1 = re.findall(r"(?<=\()(.*?)(?=::)", column_name)[0]
            label0 = re.findall(r"(?<=::)(.*?)(?=\))", column_name)[0]
            prior1 = priors[column_name][0] * 100
            cluster1 = means.iloc[attribute] * 100
            plot_data = {'group': ["cluster"] * 2 + ["all data"] * 2,
                         "distribution": [100 - cluster1, cluster1, 100 - prior1, prior1],
                         "label": [label0, label1] * 2}
            df_explanation = pd.DataFrame(plot_data)
            fig = px.bar(df_explanation, x='group', y='distribution', color='label', width=400, height=300)

        fig.update_layout(
            font_size=10,
            legend_font_size=10
        )
        fig.update_layout(margin=dict(t=0))
        figures.append(html.H6(
            [name, dbc.Badge(format(ics[cluster][attribute], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id="Cluster " + str(cluster) + ", " + column_names[attribute],
                                 figure=fig,
                                 config={
                                     'displayModeBar': False
                                 })
                       )
    return figures


def config_layout(scatter, explanation, cluster_ids, dashboard, dataset_name):
    return html.Div([
        # Top Navbar
        dbc.Row(dbc.Col(dbc.NavbarSimple(brand="ExClus", color="primary", dark=True, fluid=True, sticky="top"))),
        # Dashboard with general info
        dbc.Card(
            dbc.CardBody(
                [
                    # Dashboard
                    dbc.Row(
                        dbc.Col(
                            children=dashboard,
                            width=7,
                            id="dashboard",
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
                                                scatter
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
                                                        {'label': "Cluster " + str(i), 'value': i} for i in cluster_ids
                                                    ],
                                                    value=0
                                                ),
                                                html.Div(explanation, id="explanation", style=SIDEBAR_STYLE)
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


def render(df_data, df_data_scaled, path_to_emb_file, dataset_name):
    embedding = compute_embedding(df_data_scaled, path_to_emb_file)
    optimiser = ExclusOptimiser(df_data, df_data_scaled, embedding, name=dataset_name)
    labels, attributes, si = optimiser.optimise()
    ics = optimiser.get_ic_opt()
    priors = optimiser.get_priors()
    dls = optimiser.get_dls()

    scatter = config_scatter(embedding, labels)
    explanations = config_explanation(df_data, labels, attributes, priors, dls, ics)
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.css.config.serve_locally = False
    my_css_urls = ["https://codepen.io/rmarren1/pen/mLqGRg.css"]
    for url in my_css_urls:
        app.css.append_css({
            "external_url": url
        })

    cluster_ids = list(set(labels))
    dashboard = render_dashboard(cluster_ids, attributes, optimiser.get_total_ic_opt())
    app.layout = config_layout(scatter, explanations, cluster_ids, dashboard, dataset_name)
    return app, optimiser


def load_data(path, n_samples=2500, state=2):
    data = pd.read_csv(path)
    try:
        data_sampled = data.sample(n_samples, random_state=state, axis=0)
    except:
        data_sampled = data
    scaler = StandardScaler()
    data_scaled = copy.deepcopy(data_sampled)
    data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled[data_scaled.columns])
    return data_sampled, data_scaled
