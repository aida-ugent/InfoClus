import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import re
import plotly.io as pio
import os

from caching import from_cache


'''
def config_clustering(embedding, labels):
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
    # figure.write_html('first_figure.html', auto_open=True)
'''

def config_explanation(data, labels, attributes, priors, dls, ics, max_cluster):

    figures = []

    for cluster in range(max_cluster+1):

        cluster_data = data.iloc[np.nonzero(labels == cluster)[0], :]
        column_names = data.columns
        means = cluster_data.mean()
        stds = cluster_data.std()

        for attribute in attributes[cluster]:

            # DL = 2 so show to normal distributed curves (prior and cluster)
            if dls[attribute] == 2:
                min_val = min(means.iloc[attribute] - 4 * stds.iloc[attribute],
                              priors[attribute][0] - 4 * priors[attribute][1])
                max_val = max(means.iloc[attribute] + 4 * stds.iloc[attribute],
                              priors[attribute][0] + 4 * priors[attribute][1])
                x = np.linspace(min_val, max_val, 1000)
                epsilon = 0
                if stds.iloc[attribute] == 0:
                    epsilon = (max_val - min_val) / 100
                y_cluster = ss.norm.pdf(x, means.iloc[attribute], stds.iloc[attribute] + epsilon)
                y_prior = ss.norm.pdf(x, priors[attribute][0], priors[attribute][1])
                plot_data = {column_names[attribute]: np.concatenate((x, x)),
                             "pdf": np.concatenate((y_cluster, y_prior)),
                             'labels': ['cluster'] * 1000 + ['all data'] * 1000}
                df_explanation = pd.DataFrame(plot_data)

                fig = px.line(df_explanation, x=column_names[attribute], y="pdf", color='labels',
                              title=f'cluster {cluster} - attribute {attribute}', width=400, height=300)
            # DL = 1 and it is a binary attribute, so show 2 stacked bar plots (cluster and prior)
            else:
                column_name = column_names[attribute]
                label1 = re.findall(r"(?<=\()(.*?)(?=::)", column_name)[0]
                label0 = re.findall(r"(?<=::)(.*?)(?=\))", column_name)[0]
                prior1 = priors[attribute][0] * 100
                cluster1 = means.iloc[attribute] * 100
                plot_data = {column_names[attribute]: ["cluster"] * 2 + ["all data"] * 2,
                             "distribution": [100 - cluster1, cluster1, 100 - prior1, prior1],
                             "label": [label0, label1] * 2}
                df_explanation = pd.DataFrame(plot_data)

                fig = px.bar(df_explanation, x=column_names[attribute], y='distribution', color='label',
                             title=f'cluster {cluster} - attribute {attribute}', width=400, height=300)
            figures.append(fig)

    return figures

def painting(data_name, work_folder, file_to_painting, data):

    path_exclus_res = f'{work_folder}/{file_to_painting}'
    exclus_info = from_cache(path_exclus_res)

    data_prior = exclus_info["prior"]
    dls = exclus_info['dls']
    res_in_brief = exclus_info['res_in_brief']
    clustering = exclus_info["clustering"]
    attributes = exclus_info['attributes']
    ics = exclus_info['ic']
    max_cluster = exclus_info['maxlabel']

    # figure_clustering = config_clustering(data_emb, clustering)
    figures_explanation = config_explanation(data, clustering, attributes, data_prior, dls, ics, max_cluster)
    # config_explanation(data, clustering, attributes, data_prior, dls, ics, max_cluster)
    html_figs = []
    # html_figs.append(pio.to_html(figure_clustering, full_html=False))
    for figures_explanation in figures_explanation:
        html_figs.append(pio.to_html(figures_explanation, full_html=False))
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plotly Figures</title>
        <style>
            body {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
            }}
            .plotly-figure {{
                width: 45%;
                margin: 20px;
            }}
        </style>
    </head>
    <body>
        {res_in_brief}
        {plots}
    </body>
    </html>
    """

    res_in_brief = res_in_brief
    plots = "".join(f'<div class="plotly-figure">{fig}</div>' for fig in html_figs)

    # 将图表字符串插入到 HTML 模板中
    html_content = html_template.format(res_in_brief=res_in_brief, plots=plots)

    # Define the directory and file name
    directory = f"../data/{data_name}"  # 替换为你想要保存的目录路径
    file_name = f'{file_to_painting}.html'

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the full path
    file_path = os.path.join(directory, file_name)

    with open(file_path, "w") as f:
        f.write(html_content)

    print(f"HTML file created successfully: {file_to_painting}.html")
