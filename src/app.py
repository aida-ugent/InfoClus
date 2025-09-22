import os
from pydoc import html

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from layout import config_layout
from callbacks import register_callbacks
from dash_utils import build_infoclus, serialize_obj
from config import PROJECT_ROOT
from dash_utils import serialize_obj

data_name = 'german_socio_eco'
df_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', data_name, f'{data_name}.csv'))

infoclus_obj = build_infoclus('german_socio_eco')
infoc_para_res_dict = infoclus_obj.optimise()
labels = {'infoclus_clustering': infoc_para_res_dict['clustering']}

embeddings_load = np.load(os.path.join(PROJECT_ROOT,'data', data_name, 'cache', 'embeddings.npz'))
embeddings = {k: embeddings_load[k] for k in embeddings_load.files}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "InfoClus | A Dashboard for explainable clustering helping you understand your dataset better"
app.css.config.serve_locally = False
my_css_urls = ["https://codepen.io/rmarren1/pen/mLqGRg.css"]
for url in my_css_urls:
    app.css.append_css({
        "external_url": url
    })

app.layout = dash.html.Div(
    id='main-div',
    children=[
        dash.dcc.Store(id='infoclus_store', storage_type='memory', data=infoc_para_res_dict),
        dash.dcc.Store(id='dataset_store', storage_type='memory', data=serialize_obj(df_data)),
        dash.dcc.Store(id='embedding_store', storage_type='memory', data=serialize_obj(embeddings)),
        dash.dcc.Store(id='clustering_store', storage_type='memory', data=labels),
        dbc.Container(children=
            [
                dash.html.Header(
                    children= "InfoClus",
                    id="infoc_bar",
                    className="my-navbar",
                    style={'height': '8vh'}),
                dbc.Container(
                    fluid=True,
                    id='dashboard-content',
                    children=config_layout(infoc_para_res_dict, df_data, embeddings, labels),
                    className="my-navbar",
                    style={'height': '85vh'}
            )],
            fluid=True,
            style={'height': '100vh'}
        )]



)

register_callbacks(app)

if __name__ == "__main__":
    app.run(
        debug=True, port=8051, dev_tools_hot_reload=True, use_reloader=True
    )


