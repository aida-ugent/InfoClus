import plotly.express as px
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc

from dash import dcc
from dash import html

from ..caching import from_cache

def get_clustering(clustering: np.ndarray, embedding: np.ndarray):
    """
    :return: a plotly.express figure used to show final clustering result
    """
    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': clustering  # Classifications
    })
    fig = px.scatter(df, x='x', y='y', color='class', title='clustering')
    return fig

def config_scatter_graph(results_file: str = 'german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0', emb_name: str = 'tSNE_1'):

    infoclus = from_cache(results_file)
    clustering = infoclus["clustering"]
    embedding = infoclus['embs'][emb_name]

    graph = dcc.Graph(
        id="embedding-scatterPlot",
        figure=get_clustering(clustering, embedding)
    )

    return graph


def get_distribution(results_file: str, attribute, cluster):
    """
    :return: return distribution (kde/barplot) of given parameters
    """

    # data
    full_data = data.jloc[attribute]
    cluster_data = data.iloc[cluster, attribute]

    pass


def config_explanation(results_file: str = "german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0", cluster: int = 0):
    """
    :param results_file: file where storing infoclus result
    :return: distributions for all selected features in a cluster, default as 0
    """

    figures = [html.Br(), dbc.Alert("Contains " + format(percentage, '.2f') + ' % of data', color="info")]
    for attribute in attributes:
        figures.append(html.H6(
            [name, dbc.Badge(format(ics[cluster][attribute], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id="Cluster " + str(cluster) + ", " + column_names[attribute],
                                 figure=get_distribution(results_file, attribute, cluster),
                                 config={
                                     'displayModeBar': False
                                 }))

    pass