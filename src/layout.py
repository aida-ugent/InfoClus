import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import fastkde
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

# Performance: use WebGL scatter above this many points; subsample KDE above this
SCATTER_WEBGL_THRESHOLD = 15000
KDE_MAX_SAMPLES = 15000
# Smooth KDE curves to mimic a slightly larger bandwidth.
KDE_SMOOTH_WINDOW = 11

# Matplotlib tab20 palette, aligned with notebook plotting.
TAB20_COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
]

def _subsample_for_kde(arr: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample array for KDE to keep computation fast on large datasets."""
    n = len(arr)
    if n <= max_samples:
        return arr
    idx = rng.choice(n, size=max_samples, replace=False)
    return np.take(arr, np.sort(idx))


def _smooth_density(density: np.ndarray, window: int = KDE_SMOOTH_WINDOW) -> np.ndarray:
    """Apply a small moving-average smoother to produce more stable KDE curves."""
    y = np.asarray(density, dtype=float).ravel()
    if window <= 1 or y.size < 3:
        return y
    w = min(window, y.size if y.size % 2 == 1 else y.size - 1)
    if w < 3:
        return y
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y, kernel, mode='same')


def _cluster_color_map(clustering: np.ndarray) -> dict:
    """Map each cluster label to a stable tab20 color."""
    unique_clusters = np.unique(np.asarray(clustering))
    return {cluster: TAB20_COLORS[i % len(TAB20_COLORS)] for i, cluster in enumerate(unique_clusters)}


def get_fastkde(data_att: np.ndarray, cluster_att: np.ndarray, att_name: str, ic, cluster_color: str = '#2ca02c'):
    n_data, n_cluster = len(data_att), len(cluster_att)
    percentage = n_cluster / n_data if n_data else 0
    rng = np.random.default_rng(42)
    data_att = _subsample_for_kde(np.asarray(data_att).ravel(), KDE_MAX_SAMPLES, rng)
    cluster_att = _subsample_for_kde(np.asarray(cluster_att).ravel(), KDE_MAX_SAMPLES, rng)

    pdf_data = fastkde.pdf(data_att, var_names='d')
    grid_data = pdf_data.coords['d'].values
    density_data = _smooth_density(pdf_data.values)

    pdf_cluster = fastkde.pdf(cluster_att, var_names='c')
    grid_cluster = pdf_cluster.coords['c'].values
    density_cluster = _smooth_density(pdf_cluster.values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid_data, y=density_data, mode='lines', name=f'Data',
                             line=dict(color='blue', width=2),
                             hovertemplate='Data distr.<br>( %{x:.2f}, %{y:.2f})<extra></extra>'))
    fig.add_trace(go.Scatter(x=grid_cluster, y=density_cluster, mode='lines', name=f'cluster',
                             line=dict(color=cluster_color, width=2, dash='dot'),
                             hovertemplate='Cluster distr.<br> (%{x:.2f}, %{y:.2f})<extra></extra>'))
    fig.add_trace(go.Scatter(x=grid_cluster, y=density_cluster*percentage, fill='tozeroy', name=f'{percentage}% Overlapped by Cluster',
                             line=dict(color=cluster_color, width=1),
                             fillcolor=cluster_color,
                             opacity=0.35,
                             hovertemplate='Coverage of cluster<br> (%{x:.2f}, %{y:.2f})<extra></extra>'))

    fig.update_layout(
        xaxis=dict(
            title=att_name + "-IC-"+str(round(ic,1)),
            showline=True,
            linecolor="gray",
            linewidth=1,
            titlefont=dict(size=18),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title="KDE PDFs",
            showline=True,
            linecolor="gray",
            linewidth=1,
            titlefont=dict(size=18),
            tickfont=dict(size=14),
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=14),
    )

    return fig

def config_scatter_graph(clustering: list, embedding: np.ndarray, marker_size: float = 1, marker_opacity: float = 1.0):
    clustering = np.asarray(clustering)
    embedding = np.asarray(embedding)
    n_points = len(embedding)
    color_map = _cluster_color_map(clustering)

    if n_points > SCATTER_WEBGL_THRESHOLD:
        # WebGL scatter for large datasets (much faster than SVG)
        fig = go.Figure()
        for c in np.unique(clustering):
            mask = clustering == c
            idx = np.where(mask)[0]
            fig.add_trace(go.Scattergl(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                name=str(c),
                marker=dict(size=marker_size, opacity=marker_opacity, color=color_map[c]),
                customdata=idx.reshape(-1, 1),
                hovertemplate='class=%{fullData.name}, x=%{x:.2f}, y=%{y:.2f}<extra></extra>',
            ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
        )
        return fig

    color_map_str = {str(k): v for k, v in color_map.items()}
    class_values = np.asarray(clustering).astype(str)
    class_order = [str(k) for k in np.unique(clustering)]

    df = pd.DataFrame({
        'x': embedding[:, 0],  # X coordinates
        'y': embedding[:, 1],  # Y coordinates
        'class': pd.Categorical(class_values, categories=class_order, ordered=True),  # Classifications
        'customdata': list(range(n_points)),
    })
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='class',
        custom_data=['customdata'],
        color_discrete_map=color_map_str,
        category_orders={'class': class_order},
    )
    fig.update_traces(
        marker=dict(size=marker_size, opacity=marker_opacity),
        hovertemplate='class=%{fullData.name}, x=%{x:.2f}, y=%{y:.2f}<extra></extra>',
    )
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
    att_names = df_data.columns
    ics_cluster = np.array(infoc_para_res['ic_opt'][cluster_label])
    cluster_color = _cluster_color_map(np.asarray(infoc_para_res['clustering'])).get(
        cluster_label,
        TAB20_COLORS[int(cluster_label) % len(TAB20_COLORS)],
    )
    for att_id in infoc_para_res['attributes_opt'][cluster_label]:
        data_att = scaled_data[:, att_id]
        cluster_att = cluster[:, att_id]
        att_name = att_names[att_id]
        if infoc_para_res['global_arr_type'] == 'categorical':
            # fig = get_barchart(infoclus, att_id, cluster_label, att_name)
            pass
        elif infoc_para_res['global_arr_type'] == 'numeric':
            fig_fastkde = get_fastkde(data_att, cluster_att, att_name, ics_cluster[att_id], cluster_color=cluster_color)
        else:
            print('unsupported attribute type for visualization:', infoc_para_res['global_arr_type'])

        # figures.append(html.H6([att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(id=f"Cluster {cluster_label}, {att_name}",
                                 figure=fig_fastkde,
                                 style = {'width': '100%', 'height': '40%'},
                                 config = {'responsive': True}
                                 )
                       )
    return figures

def config_selected_explanations(infoc_para_res: dict = None, df_data: pd.DataFrame=None, selected_idxes=None, ics_cluster=None, attributes=None):
    """Returns (title_children, body_children) for the KDE section."""
    title_style = {'fontWeight': '600', 'color': '#555', 'fontSize': '2rem'}
    default_title = html.H5("Explanation for the selected region", className="mb-0", style=title_style)
    default_body = html.P("Select points in the scatter plot above to see explanations given by KDEs.", className="text-muted mb-0")

    if selected_idxes is None:
        return default_title, default_body

    scaled_data = np.array(infoc_para_res['scaled_data'])
    cluster = scaled_data[selected_idxes]
    percentage = len(selected_idxes) / df_data.shape[0] * 100

    kde_title = html.H5(
        f"Explanation for the selected region (contain {format(percentage, '.2f')}% of data)",
        className="mb-0",
        style=title_style,
    )

    figures = []
    att_names = df_data.columns
    selected_color = TAB20_COLORS[0]

    for att_id in attributes:
        data_att = scaled_data[:, att_id]
        cluster_att = cluster[:, att_id]
        att_name = att_names[att_id]
        if infoc_para_res['global_arr_type'] == 'categorical':
            # fig = get_barchart(infoclus, att_id, cluster_label, att_name)
            pass
        elif infoc_para_res['global_arr_type'] == 'numeric':
            fig_fastkde = get_fastkde(data_att, cluster_att, att_name, ics_cluster[att_id], cluster_color=selected_color)
        else:
            print('unsupported attribute type for visualization:', infoc_para_res['global_arr_type'])
        # figures.append(html.H6(
        #     [att_name, dbc.Badge(format(ics_cluster[att_id], '.1f') + " IC", color="success", className="ml-1")]))
        figures.append(dcc.Graph(
                                 figure=fig_fastkde,
                                 style={
                                     'height': '100%',
                                     'minHeight': '280px',
                                     'flex': 1,
                                     'minWidth': '320px',
                                 },
                                 config={'responsive': True}
                                 )
                       )

    body = html.Div(
        figures,
        style={
            'display': 'flex',
            'flexWrap': 'nowrap',
            'height': '100%',
            'minHeight': '280px',
            'minWidth': 'min-content',
        },
    )
    return kde_title, body

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
        {'label': 'recalculate in 30 s', 'value': '5'},
        {'label': 'recalculate in 60 s', 'value': '6'}
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

    first_label_key = list(labels.keys())[0] if labels else 'InfoClus'
    return html.Div(
        id='layout',
        style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'},
        children=[
            dcc.Store(id='clustering-value-store', data=first_label_key),
            dcc.Store(id='embedding-value-store', data=main_emb_name),
            dcc.Store(id='cluster-select-value-store', data=cluster_id),
            dbc.Row(
                id='layout-row',
                style={'flex': 1, 'minHeight': 0, 'alignItems': 'stretch'},
                children=[
                    dbc.Col(
                        xs=12,
                        sm=3,
                        md=3,
                        id='selection-panel',
                        style={'display': 'flex', 'flexDirection': 'column', 'minHeight': 0},
                        children=dbc.Card(
                        dbc.CardBody([
                        html.Div(
                            children=[
                                html.P(
                                    'Welcome to InfoClus, '
                                    'a new clustering method that also explains its clusters. '
                                    'Play with existed datasets or import your own dataset!',
                                    id='welcome-block',
                                    className='mb-4',
                                    style={'fontSize': '2rem', 'lineHeight': 1.5, 'color': '#555'},
                                ),
                                dbc.Card(
                                    dbc.CardBody(children=[
                                        dcc.Upload(
                                            id='import-dataset',
                                            children=html.Button('Upload Dataset', style={'fontSize': '1.5rem'}),
                                        ),
                                        dcc.Upload(
                                            id='import-embedding',
                                            children=html.Button('Upload Embedding', style={'fontSize': '1.5rem'})
                                        ),
                                    ], style={'padding': '0.5rem'}),
                                    color='light',
                                    className='mb-4',
                                    style={'flexShrink': 0},
                                ),
                                dbc.Card(
                                    dbc.CardBody(children=[
                                        html.Label('Dataset', className='form-label', style={'fontSize': '2rem', 'marginBottom': '0.25rem'}),
                                        dcc.Dropdown(
                                            options=[{'label': dataset, 'value': dataset} for dataset in get_datasets()],
                                            value=dataset_name,
                                            id='dataset-select',
                                            style={'width': '100%'},
                                        ),
                                        html.P(
                                            f"# instances: {df_data.shape[0]}, # dimensions: {df_data.shape[1]}",
                                            id='dataset-stats',
                                            className='text-muted small mb-0',
                                            style={'marginTop': '0.35rem', 'fontSize': '2rem'},
                                        ),
                                        html.Label('Embedding', className='form-label', style={'fontSize': '2rem', 'marginTop': '0.5rem', 'marginBottom': '0.25rem'}),
                                        dcc.Dropdown(
                                            options=[{'label': emb_name, 'value': emb_name} for emb_name in get_embeddings_keys(dataset_name)],
                                            value=main_emb_name,
                                            id='embedding-select',
                                            style={'width': '100%'},
                                        ),
                                    ], style={'padding': '0.5rem'}),
                                    color='light',
                                    className='mb-4',
                                    style={'flexShrink': 0},
                                ),
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.H6('Hyper-parameters', className='card-title', style={'fontSize': '2rem', 'marginBottom': '0.35rem'}),
                                                    html.Label('alpha', className='small text-muted', style={'fontSize': '2rem', 'display': 'block', 'marginBottom': '0.2rem'}),
                                                    dcc.Slider(
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
                                                ],
                                                style={'flexShrink': 0},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label('beta', className='small text-muted', style={'fontSize': '2rem', 'display': 'block', 'marginBottom': '0.2rem'}),
                                                    dcc.Slider(
                                                        id='beta-slider',
                                                        min=1,
                                                        max=2,
                                                        step=0.1,
                                                        marks={1: {'label': '1'}, 2: {'label': '2'}},
                                                        value=infoc_para_res['beta'],
                                                        tooltip={"always_visible": True, 'placement': 'bottom'},
                                                    ),
                                                ],
                                                style={'flexShrink': 0},
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Label('min_att', className='small text-muted', style={'fontSize': '2rem', 'marginBottom': '0.2rem'}), dcc.Input(type="number", value=2, step=1, min=0, max=10, id='min-att-input', className='form-control form-control-sm')], width=6),
                                                    dbc.Col([html.Label('max_att', className='small text-muted', style={'fontSize': '2rem', 'marginBottom': '0.2rem'}), dcc.Input(type="number", value=5, step=1, min=3, max=10, id='max-att-input', className='form-control form-control-sm')], width=6),
                                                ],
                                                className='g-2',
                                                style={'flexShrink': 0},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label('Recalc runtime', className='small text-muted', style={'fontSize': '2rem', 'display': 'block', 'marginBottom': '0.2rem'}),
                                                    dcc.Dropdown(
                                                        value='1',
                                                        options=get_runtime_dropdown_items(),
                                                        id='recalc-hyperparameters',
                                                        style={'width': '100%'},
                                                    ),
                                                ],
                                                style={'flexShrink': 0},
                                            ),
                                        ],
                                        style={
                                            'flex': 1,
                                            'display': 'flex',
                                            'flexDirection': 'column',
                                            'justifyContent': 'start',
                                            'padding': '0.5rem',
                                            'minHeight': 0,
                                        },
                                    ),
                                    color='light',
                                    className='mb-4',
                                    style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'minHeight': 0},
                                ),
                            ],
                            style={
                                'flex': 1,
                                'display': 'flex',
                                'flexDirection': 'column',
                                'minHeight': 0,
                                'overflow': 'hidden',
                            },
                        )],
                            style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'minHeight': 0, 'overflow': 'hidden'},
                        ),
                            style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'minHeight': 0, 'fontFamily': '"Inter", sans-serif'},
                            className='h-100 infoc-column-card',
                        )
                    ),
                dbc.Col(
                    xs=12,
                    sm=6,
                    md=6,
                    id = 'clustering-panel',
                    className='h-100',
                    style={'display': 'flex', 'flexDirection': 'column', 'minHeight': 0},
                    children=dbc.Card(
                        dbc.CardBody(
                            [
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                dbc.DropdownMenu(
                                                                    id='clustering-dropdown-menu',
                                                                    label=first_label_key,
                                                                    children=[
                                                                        dbc.DropdownMenuItem(
                                                                            k,
                                                                            id={'type': 'cluster-option', 'index': k},
                                                                        )
                                                                        for k in labels.keys()
                                                                    ]
                                                                    + [
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
                                                                    ],
                                                                    color='light',
                                                                    size='sm',
                                                                    className='scatter-control-btn me-1',
                                                                ),
                                                                dbc.Tooltip(
                                                                    'Select or upload labels',
                                                                    target='clustering-labels-hint',
                                                                    placement='top',
                                                                ),
                                                            ],
                                                            id='clustering-labels-hint',
                                                            style={'display': 'inline-flex', 'alignItems': 'center'},
                                                        ),
                                                        width='auto',
                                                        className='d-flex align-items-center',
                                                    ),
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                dbc.DropdownMenu(
                                                                    id='embedding-dropdown-menu-show',
                                                                    label=main_emb_name,
                                                                    children=[
                                                                        dbc.DropdownMenuItem(
                                                                            emb_name,
                                                                            id={'type': 'embedding-option', 'index': emb_name},
                                                                        )
                                                                        for emb_name in get_embeddings_keys(dataset_name)
                                                                    ],
                                                                    color='light',
                                                                    size='sm',
                                                                    className='scatter-control-btn px-1',
                                                                ),
                                                                dbc.Tooltip(
                                                                    'Select embedding to show',
                                                                    target='embedding-show-hint',
                                                                    placement='top',
                                                                ),
                                                            ],
                                                            id='embedding-show-hint',
                                                            style={'display': 'inline-flex', 'alignItems': 'center'},
                                                        ),
                                                        width='auto',
                                                        className='px-1 d-flex align-items-center',
                                                    ),
                                                    dbc.Col(
                                                        html.Div(
                                                            dbc.Button(
                                                                'Point style',
                                                                id='scatter-point-style-toggle',
                                                                color='light',
                                                                size='sm',
                                                                outline=False,
                                                                className='scatter-control-btn ms-1',
                                                            ),
                                                            style={'display': 'inline-flex', 'alignItems': 'center'},
                                                        ),
                                                        width='auto',
                                                        className='d-flex align-items-center',
                                                    ),
                                                ],
                                                className='g-1 align-items-center justify-content-center',
                                            ),
                                            width=12,
                                        ),
                                        dbc.Col(
                                            dbc.Collapse(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col([
                                                                        html.Label('Point size', className='small mb-0'),
                                                                        dcc.Slider(
                                                                            id='scatter-point-size',
                                                                            min=0.5,
                                                                            max=15,
                                                                            step=0.5,
                                                                            value=4,
                                                                            marks={0.5: '0.5', 15: '15'},
                                                                            tooltip={'placement': 'bottom', 'always_visible': True},
                                                                        ),
                                                                    ], width=6),
                                                                    dbc.Col([
                                                                        html.Label('Transparency', className='small mb-0'),
                                                                        dcc.Slider(
                                                                            id='scatter-point-opacity',
                                                                            min=0.1,
                                                                            max=1,
                                                                            step=0.05,
                                                                            value=1,
                                                                            marks={0.1: '0.1', 1: '1'},
                                                                            tooltip={'placement': 'bottom', 'always_visible': True},
                                                                        ),
                                                                    ], width=6),
                                                                ],
                                                                className='g-2 align-items-end',
                                                            ),
                                                        ],
                                                        className='py-2',
                                                    ),
                                                    className='mb-2',
                                                ),
                                                id='scatter-point-style-collapse',
                                                is_open=False,
                                            ),
                                            width=12,
                                        ),
                                    ],
                                    className='mb-2 align-items-center g-1',
                                    style={'flexShrink': 0},
                                ),
                                html.Div(
                                    style={
                                        'flex': '0 0 60%',
                                        'minHeight': 0,
                                        'display': 'flex',
                                        'flexDirection': 'column',
                                    },
                                    children=[
                                        html.Div(
                                            id='scatter-plot-title-container',
                                            style={
                                                'width': '100%',
                                                'textAlign': 'center',
                                                'flexShrink': 0,
                                                'paddingBottom': '0.5rem',
                                            },
                                            children=[
                                                html.Span(
                                                    id='scatter-plot-title',
                                                    children=f'{dataset_name} embedding',
                                                    style={
                                                        'fontSize': '2rem',
                                                        'fontWeight': '600',
                                                        'color': '#555',
                                                    },
                                                ),
                                                dbc.Tooltip(
                                                    id='scatter-plot-tooltip',
                                                    target='scatter-plot-title',
                                                    placement='top',
                                                    children=f'Embedding computed on {main_emb_name}, cluster labels are computed/given by InfoClus',
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={
                                                'position': 'relative',
                                                'flex': 1,
                                                'minHeight': 0,
                                                'overflow': 'hidden',
                                                'minWidth': 0,
                                            },
                                            children=[
                                                dcc.Graph(
                                                    id="embedding-scatterPlot",
                                                    figure=config_scatter_graph(
                                                        labels['InfoClus'],
                                                        embeddings[main_emb_name],
                                                        marker_size=4,
                                                        marker_opacity=1.0,
                                                    ),
                                                    style={'height': '100%', 'width': '100%'},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # dcc.Markdown(
                                #     r"$R_{\alpha,\beta}(\mathcal{C}, \mathcal{E}) = \frac{\sum_{i=1}^r{\sum_{j=1}^{|e_i|}{I_i^j}}}{\alpha + (\sum_{i=1}^r{\sum_{j=1}^{|e_i|}{|a_i^j|}})^\beta}$ is ...",
                                #     mathjax=True
                                # ),
                                html.Div(
                                    style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'flex': '0 0 30%', 'minHeight': 0},
                                    children=[
                                        html.Div(
                                            id='selected-explanation-title',
                                            children=config_selected_explanations()[0],
                                            style={'width': '100%', 'textAlign': 'center', 'flexShrink': 0, 'paddingBottom': '0.5rem'},
                                        ),
                                        html.Div(
                                            className='kde-scroll-hide-horizontal',
                                            style={
                                                'flex': 1,
                                                'minHeight': '280px',
                                                'display': 'flex',
                                                'flexDirection': 'column',
                                                'overflowX': 'auto',
                                                'overflowY': 'auto',
                                            },
                                            children=[
                                                dbc.Row(
                                                    dbc.Col(
                                                        id='selected-explanation',
                                                        children=config_selected_explanations()[1],
                                                        style={
                                                            'display': 'flex',
                                                            'minWidth': 'min-content',
                                                            'height': '100%',
                                                        },
                                                    ),
                                                    style={'flex': 1, 'minHeight': 0, 'height': '100%'},
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                            style={'flex': 1, 'minHeight': 0, 'display': 'flex', 'flexDirection': 'column', 'overflowY': 'auto', 'overflowX': 'hidden'},
                            ),
                            ],
                            style={'flex': 1, 'minHeight': 0, 'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'},
                        ),
                    style={'height': '100%', 'display': 'flex', 'flexDirection': 'column', 'minHeight': 0},
                    className='h-100 infoc-column-card',
                ),
                ),
                dbc.Col(
                    xs=12,
                    sm=3,
                    md=3,
                    id='explanation-panel',
                    className='h-100',
                    children=dbc.Card(
                        dbc.CardBody([
                            html.Div(
                                id='cluster-select-wrapper',
                                children=[
                                    html.Span("Explanation for ", style={'fontSize': '2rem', 'fontWeight': '500'}),
                                    dbc.DropdownMenu(
                                        id='cluster-select-menu',
                                        label=f"Cluster {cluster_id}",
                                        children=[
                                            dbc.DropdownMenuItem(
                                                "Cluster " + str(i),
                                                id={'type': 'cluster-select-option', 'index': i},
                                            )
                                            for i in range(infoc_para_res['count_clusters'])
                                        ],
                                        color='light',
                                        size='sm',
                                        className='scatter-control-btn me-1',
                                    ),
                                    dbc.Button(
                                        "LLM",
                                        id='llm-button',
                                        color='light',
                                        size='sm',
                                        className='scatter-control-btn ms-2',
                                    ),
                                    dbc.Tooltip(
                                        id='cluster-select-tooltip',
                                        target='cluster-select-wrapper',
                                        placement='top',
                                    ),
                                ],
                                style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '0.25rem', 'width': '100%'},
                            ),
                            dbc.Row(
                                id='explanation',
                                children=config_explanations(infoc_para_res, df_data, cluster_id),
                                style={'height': '70vh', 'overflowY': 'auto'},
                            ),
                        ]),
                        style={'height': '100%', 'overflowY': 'auto'},
                        className='h-100 infoc-column-card',
                    ),
                ),
            ],
        ),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Configure LLM API")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.CardGroup([
                        dbc.Label("API Provider", html_for='llm-provider-input', style={'fontWeight': '500'}),
                        dcc.Dropdown(
                            id='llm-provider-input',
                            options=[
                                {'label': 'OpenAI', 'value': 'openai'},
                                {'label': 'Google', 'value': 'google'},
                                {'label': 'Anthropic', 'value': 'anthropic'},
                                {'label': 'Custom', 'value': 'custom'},
                            ],
                            placeholder='Select API provider',
                            style={'width': '100%'},
                        ),
                    ]),
                    dbc.CardGroup([
                        dbc.Label("API Key", html_for='llm-api-key-input', style={'fontWeight': '500'}),
                        dcc.Input(
                            id='llm-api-key-input',
                            type='password',
                            placeholder='Enter your API key',
                            className='form-control',
                            style={'width': '100%'},
                        ),
                    ]),
                    dbc.CardGroup([
                        dbc.Label("Model Name", html_for='llm-model-input', style={'fontWeight': '500'}),
                        dcc.Input(
                            id='llm-model-input',
                            type='text',
                            placeholder='e.g., gpt-4, claude-3',
                            className='form-control',
                            style={'width': '100%'},
                        ),
                    ]),
                    dbc.CardGroup([
                        dbc.Label("API Endpoint (Optional)", html_for='llm-endpoint-input', style={'fontWeight': '500'}),
                        dcc.Input(
                            id='llm-endpoint-input',
                            type='text',
                            placeholder='Custom endpoint URL',
                            className='form-control',
                            style={'width': '100%'},
                        ),
                    ]),
                    dbc.CardGroup([
                        dbc.Label("Task (Optional)", html_for='llm-task-input', style={'fontWeight': '500'}),
                        dcc.Textarea(
                            id='llm-task-input',
                            placeholder='Describe the task for the LLM (optional)',
                            className='form-control',
                            style={'width': '100%', 'minHeight': '100px', 'fontSize': '1.5rem'},
                        ),
                    ]),
                ]),
                html.Br(),
                dbc.Button("Create prompt", id='llm-create-prompt-btn', color='primary'),
                html.Hr(),
                html.Div(
                    id='llm-prompt-output',
                    style={'whiteSpace': 'pre-wrap', 'fontSize': '1.4rem', 'color': '#444'},
                ),
                dbc.Button(
                    "Query LLM",
                    id='llm-query-btn',
                    color='secondary',
                    style={'display': 'none', 'marginTop': '0.75rem'},
                ),
                html.Div(
                    id='llm-query-output',
                    style={'whiteSpace': 'pre-wrap', 'fontSize': '1.35rem', 'color': '#333', 'marginTop': '0.75rem'},
                ),
                dbc.Button(
                    "update infoclus labels",
                    id='llm-update-labels-btn',
                    color='success',
                    style={'marginTop': '0.75rem'},
                ),
            ]),
        ],
        id='llm-modal',
        is_open=False,
        ),
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
