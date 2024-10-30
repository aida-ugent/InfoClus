import plotly.express as px
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

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


def get_kde(cluster_att: np.ndarray, data_att: np.ndarray, att_name: str):
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

def get_kdes(data: np.ndarray,
             cluster_label: int, clustering: np.ndarray, attributes: list,
             att_names: pd.DataFrame):
    """

    :param data:
    :param cluster_label:
    :param clustering:
    :param attributes:
    :param att_names:
    :return: list of go.Figure objects that show the kde plots of cluster with label cluster_label
    """
    # todo: finish this function first, then get a dash webpage asap
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