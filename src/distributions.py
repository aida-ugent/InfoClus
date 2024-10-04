import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from sklearn.neighbors import KernelDensity
from matplotlib.backends.backend_pdf import PdfPages


def kde(cluster: np.ndarray, dataset: np.ndarray, cluster_label: int, attribute_n: str,
        kernel: str, bandwidth_c: float = 0.1, bandwidth_a: float = 0.1, first: bool = False):

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
            label="{0}".format(data_name),
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

    if first and cluster_label == 1:
        ax.legend(loc="best", fontsize = 15)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(bottom=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.title(f'Cluster {cluster_label} - {attribute_n}', fontsize = 20)
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


def categorical1(Data: pd.DataFrame, attributes_chosen: list, clusters_distribution: dict,
                all_data_distribution: pd.DataFrame, res_in_brief: str, output: str):

    # from matplotlib import colormaps

    value_names_of_attributes = []
    for col in Data.columns:
        Data.loc[:, col] = Data[col].fillna('NaN')
        value_names_of_attributes.append(Data[col].factorize()[1])

    lens_of_attributes = [len(value_names) for value_names in value_names_of_attributes]
    color_data = cm.get_cmap('tab20')(np.linspace(0, 1, max(lens_of_attributes)))

    dict = {'Creator': 'My software', 'Author': 'Me', 'Keywords': res_in_brief}
    op = PdfPages(output, metadata=dict)
    legend = True
    for cluster_index, atts_cluster in enumerate(attributes_chosen):
        cmap = plt.get_cmap('tab10')
        color_cluster = cmap.colors[cluster_index]
        for att in atts_cluster:

            plt.figure()

            count_bar = 2
            # ind = np.arange(count_bar)
            width = 0.3
            labels = value_names_of_attributes[att]
            ind1 = np.arange(len(labels))
            ind2 = [x + width for x in ind1]

            len_att = lens_of_attributes[att]
            # dist_pre_cluster_att = clusters_distribution[cluster_index][0].iloc[:len_att, att]
            # dist_prior_per_att = all_data_distribution.iloc[:len_att, att]
            dist_pre_cluster_att = pd.Series(clusters_distribution[cluster_index][0].iloc[:len_att, att].values, index=labels)
            dist_prior_per_att = pd.Series(all_data_distribution.iloc[:len_att, att].values, index=labels)
            sorted_dist_pre_cluster_att = dist_pre_cluster_att.sort_values(ascending=False)
            sorted_dist_prior_per_att = dist_prior_per_att.loc[sorted_dist_pre_cluster_att.index]
            sorted_labels = sorted_dist_pre_cluster_att.index

            # plot bars in stack manner
            plt.bar(ind1, sorted_dist_pre_cluster_att,
                     color=color_cluster, width=width, label=f'cluster {cluster_index}')
            plt.bar(ind2, sorted_dist_prior_per_att,
                    color='darkgray', width=width, label='all data')

            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=15)

            plt.ylabel('Distribution', fontsize=15)
            plt.xlabel(clusters_distribution[cluster_index][0].columns[att], fontsize=15)
            # plt.title(
            #     f'Cluster {cluster_index} - {clusters_distribution[cluster_index][0].columns[att]}', fontsize = 20)

            sorted_reallabels = true_labels(sorted_labels, clusters_distribution[cluster_index][0].columns[att])
            plt.xticks(ind1, sorted_reallabels, rotation=40, fontsize = 15)
            if legend:
                plt.legend(loc='best', fontsize = 15)
            # legend = False
            plt.tight_layout()
            plt.savefig(op, format='pdf', bbox_inches='tight')

    op.close()

def true_labels(abbreviate_labels: pd.Index, attribute_name: str):

    # import sys
    # sys.path.insert(1, '../data/mushroom_binary')

    with open('names.txt', 'r') as file:
        # Read all rows line by line
        rows = file.readlines()
            # Process each row (strip removes extra spaces and newlines)

    label_dict = {}
    for row in rows:
        row = row[:-1]
        row = row.split(':')
        att = row[0]
        labels = row[1].split(',')
        Dict = {}
        for label in labels:
            abbreF = label.split('=')[0]
            abbreT = label.split('=')[1]
            Dict[abbreT] = abbreF
        label_dict[att] = Dict

    true_labels = []
    att_abbre = label_dict[attribute_name]
    for abbreviate_label in abbreviate_labels:
        if abbreviate_label in att_abbre:
            true_labels.append(att_abbre.get(abbreviate_label))
        else:
            true_labels.append('missing')
    # Create an index where the key is the item and the value is the position in the list
    true_labels = {item: idx for idx, item in enumerate(true_labels)}

    return true_labels
