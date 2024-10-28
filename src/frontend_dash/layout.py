import dash_bootstrap_components as dbc
from dash import html
import dash_core_components as dcc

from charts import config_scatter_graph, config_explanation

def get_scatter_graph():
    """
    This function plots the clustering of  points in embedding space.
    :return:
    """
    pass

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

def get_explanation():
    """
    This function visualizes explanations by showing feature distributions of cluster and the full dataset
    :return:
    """
    pass


def config_layout():
    return dbc.Container([
        # Top Navbar
        dbc.Row(dbc.Col(dbc.NavbarSimple(brand="InfoClus", color="primary", dark=True, fluid=True, sticky="top"))),
        # Dashboard with general info
        dbc.Card(
            dbc.CardBody(
                [
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
                                                config_scatter_graph(),
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
                                                html.Div(config_explanation(), id="explanation", style=SIDEBAR_STYLE)
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
    ],fluid=True
    )