import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from sklearn.neighbors import KernelDensity
from matplotlib.backends.backend_pdf import PdfPages


def kde(cluster: np.ndarray, dataset: np.ndarray, cluster_label: int, attribute_n: str,
        kernel: str, bandwidth_c: float = 0.1, bandwidth_a: float = 0.1):

    min_x = min(min(cluster), min(dataset))[0]
    max_x = max(max(cluster), max(dataset))[0]
    X_plot = np.linspace(min_x, max_x, 1000)[:, np.newaxis]

    fig, ax = plt.subplots()
    colors = ["darkorange", "navy", "cornflowerblue"]
    datas = [dataset, cluster]
    data_names = ['Distribution of full data', 'Part covered by cluster',]
    bandwidths = [bandwidth_a, bandwidth_c]
    lw = 2

    for color, data, data_name, bandwidth in zip(colors, datas, data_names, bandwidths):

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
        log_dens = kde.score_samples(X_plot)
        ax.plot(
            X_plot[:, 0],
            (data.shape[0]/dataset.shape[0]) * np.exp(log_dens),
            color=color,
            lw=lw,
            linestyle="-",
            label="'{0}'".format(data_name),
        )
        ax.fill_between(
            X_plot[:, 0],
            (data.shape[0]/dataset.shape[0]) * np.exp(log_dens),
            color=color,  # 填充颜色与曲线相同
            alpha=0.3  # 填充透明度
        )
    ax.plot(
        X_plot[:, 0],
        np.exp(kde.score_samples(X_plot)),
        color=color,
        lw=lw,
        linestyle="--",
        label='Distribution of cluster',
    )
    ax.legend(loc="best")
    ax.set_xlim(min_x, max_x)
    plt.title(f'cluster {cluster_label} - {attribute_n}')
    # plt.show()

    return plt


def bar(cluster, all_data, attribute_n, attribute_v, cluster_label):

    count_bar = len(cluster)
    ind = np.arange(count_bar)
    width = 0.5
    labels = ('cluster', 'all data')

    fig = plt.subplots()
    p1 = plt.bar(ind, cluster, width)
    p2 = plt.bar(ind, all_data, width,
                 bottom=cluster)

    plt.ylabel('Distribution')
    plt.title(f'cluster {cluster_label} - {attribute_n}')
    plt.xticks(ind, labels)
    plt.legend((p1[0], p2[0]), attribute_v, loc = 'upper center')
    # plt.show()
    return plt


def categorical(Data: pd.DataFrame, attributes_chosen: list, clusters_distribution: dict, all_data_distribution: pd.DataFrame, res_in_brief: str, output: str):

    value_names_of_attributes = []
    for col in Data.columns:
        Data.loc[:, col] = Data[col].fillna('NaN')
        value_names_of_attributes.append(Data[col].factorize()[1])

    lens_of_attributes = [len(value_names) for value_names in value_names_of_attributes]
    color_data = cm.get_cmap('tab20')(np.linspace(0, 1, max(lens_of_attributes)))

    dict = {'Creator': 'My software', 'Author': 'Me', 'Keywords': res_in_brief}
    op = PdfPages(output, metadata=dict)
    for cluster_index, atts_cluster in enumerate(attributes_chosen):
        for att in atts_cluster:

            plt.figure()

            count_bar = 2
            ind = np.arange(count_bar)
            width = 0.5
            labels = ['cluster', 'all data']

            len_att = lens_of_attributes[att]
            dist_pre_cluster_att = clusters_distribution[cluster_index][0].iloc[:len_att, att]
            dist_prior_per_att = all_data_distribution.iloc[:len_att, att]

            # plot bars in stack manner
            for value_index in range(len_att):

                if value_index == 0:
                    bottom_data = np.array([0, 0])
                else:
                    bottom_data = np.array([sum(dist_pre_cluster_att[:value_index]), sum(dist_prior_per_att[:value_index])])

                plt.bar(labels, [dist_pre_cluster_att[value_index], dist_prior_per_att[value_index]],
                        bottom = bottom_data, color= color_data[value_index], width=width,
                        label = value_names_of_attributes[att][value_index])

            plt.ylabel('Distribution')
            plt.title(f'cluster {cluster_index} - points: {clusters_distribution[cluster_index][1]} - {clusters_distribution[cluster_index][0].columns[att]}')
            plt.xticks(ind, labels)
            plt.legend(loc = 'upper center')
            plt.savefig(op, format='pdf')

    op.close()