import dash
from dash import html
import dash_bootstrap_components as dbc

from layout import config_layout
from callbacks import register_callbacks


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "InfoClus | A Dashboard for explainable clustering helping you understand your dataset better"
app.css.config.serve_locally = False
my_css_urls = ["https://codepen.io/rmarren1/pen/mLqGRg.css"]
for url in my_css_urls:
    app.css.append_css({
        "external_url": url
    })

app.layout = html.Div(config_layout())
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(
        debug=True, port=8051, dev_tools_hot_reload=True, use_reloader=True
    )
