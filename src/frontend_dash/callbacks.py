from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import callback_context

import layout


def register_callbacks(app):
    @app.callback(
        [Output("scatter", "figure"),
         Output("cluster-select", "options"),
         Output("cluster-select", "value")],
        [Input("recalc-hyperparameters", "n_clicks")],
        [State("dataset-select", "value"),
         State("embedding-select", "value"),
         State("alpha-slider", "value"),
         State("beta-slider", "value"),
         State("runtime-slider", "value"),
         State("minAtt-slider", "value")],
        prevent_initial_call=True
    )
    def optimise_different_hyperparameters(change_hyper, alpha, beta, runtime, minAtt):

        # should layout has parameters? Edith does not have, but I need. done

        # todo: call infoclus computation based on dataset, embedding & alpha, beta, runtime, minAtt
        # todo: store infoclus result to h5
        # todo: call functions to get output components based on adata

        pass