import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KernelDensity


def kde(cluster: np.ndarray, dataset: np.ndarray, cluster_label: int, attribute_n: str,
        kernel: str, bandwidth_c: float = 0.1, bandwidth_a: float = 0.1):

    # min - max
    min_x = int(min(min(cluster), min(dataset)))
    max_x = int(max(max(cluster), max(dataset)))
    X_plot = np.linspace(min_x, max_x, 1000)[:, np.newaxis]
    # # quantile
    # min_x = min(np.percentile(cluster, 25), np.percentile(dataset, 25))
    # max_x = max(np.percentile(cluster, 75), np.percentile(dataset, 75))
    # iqr = max_x - min_x
    # X_plot = np.linspace(min_x-iqr/4, max_x+iqr/4, 1000)[:, np.newaxis]

    fig, ax = plt.subplots()
    colors = ["darkorange", "navy", "cornflowerblue"]
    datas = [cluster, dataset]
    data_names = ['cluster', 'all data']
    bandwidths = [bandwidth_c, bandwidth_a]
    lw = 2

    for color, data, data_name, bandwidth in zip(colors, datas, data_names, bandwidths):

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
        log_dens = kde.score_samples(X_plot)
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens),
            color=color,
            lw=lw,
            linestyle="-",
            label="'{0}'".format(data_name),
        )
    ax.legend(loc="upper left")
    ax.set_xlim(min_x, max_x)
    plt.title(f'cluster {cluster_label} - {attribute_n}')
    # plt.show()

    return plt


# N = 100
# X = np.concatenate(
#     (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
# )[:, np.newaxis]
# Y = np.concatenate(
#     (np.random.normal(1, 3, int(0.3 * N)), np.random.normal(5, 2, int(0.7 * N)))
# )[:, np.newaxis]
# plt = kde(X, Y, 0, 'test')
# plt.show()


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


# att_v1 = (20, 40)
# att_v2 = (80, 60)
# attribute_n = 'gender'
# attribute_v = ('male', 'female')
#
# bar(att_v1, att_v2, attribute_n, attribute_v, 0)
# plt.show()

