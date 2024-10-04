import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import re
import plotly.io as pio
import os
import mpld3

from caching import from_cache
from distributions import kde, bar, categorical,categorical1
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

kernels = ["gaussian", "tophat", "epanechnikov"]
kernel = kernels[0]


def figures(data, labels, attributes,dls, ics, max_cluster, res_in_brief, output):
    figures = []
    dict = {'Creator': 'My software', 'Author': 'Me', 'Keywords': res_in_brief}
    op = PdfPages(output, metadata = dict)

    first = True
    for cluster in range(max_cluster+1):
        cluster_data = data.iloc[np.nonzero(labels == cluster)[0], :]
        column_names = data.columns
        for attribute in attributes[cluster]:

            # DL = 2 so show to normal distributed curves (prior and cluster)
            if dls[attribute] == 2:
                column_name = column_names[attribute]
                c_data = cluster_data[column_name].values.reshape(-1, 1)
                a_data = data[column_name].values.reshape(-1, 1)
                # if kernel == 'guassian':
                q_c1 = np.percentile(c_data, 25)
                q_c3 = np.percentile(c_data, 75)
                iqr_c = q_c3 - q_c1
                q_a1 = np.percentile(a_data, 25)
                q_a3 = np.percentile(a_data, 75)
                iqr_a = q_a3 - q_a1
                min_c = min(np.std(c_data), iqr_c/1.34+0.00001)
                min_a = min(np.std(a_data), iqr_a/1.34+0.00001)
                bandwidth_c = 0.9 * min_c * c_data.shape[0] ** (-0.2)
                bandwidth_a = 0.9 * min_a * a_data.shape[0] ** (-0.2)
                bandwidth = max(bandwidth_a, bandwidth_c)
                fig = kde(c_data, a_data, cluster, column_name,
                          kernel, bandwidth, bandwidth, first)
                if cluster == 1:
                    first = False
                fig.savefig(op, format='pdf')

            # DL = 1 and it is a binary attribute, so show 2 stacked bar plots (cluster and prior)
            else:
                column_name = column_names[attribute]
                label1 = re.findall(r"(?<=\()(.*?)(?=::)", column_name)[0]
                label0 = re.findall(r"(?<=::)(.*?)(?=\))", column_name)[0]
                att_label1 = [sum(cluster_data[column_name]==1)/len(cluster_data[column_name]),
                              sum(data[column_name]==1)/len(data[column_name])]
                att_label0 = [1-x for x in att_label1]
                fig = bar(att_label1, att_label0, column_name, (label1, label0), cluster)
                fig.savefig(op, format='pdf')

            figures.append(fig)
    op.close()

def painting(work_folder, file_to_painting, data, output):

    path_exclus_res = f'{work_folder}/{file_to_painting}'
    exclus_info = from_cache(path_exclus_res)

    dls = exclus_info['dls']
    res_in_brief = exclus_info['res_in_brief']
    clustering = exclus_info["clustering"]
    attributes = exclus_info['attributes']
    ics = exclus_info['ic']
    max_cluster = exclus_info['maxlabel']
    binary_att_length = exclus_info['binary_att_length']

    if binary_att_length != None:
        figures(data, clustering, attributes, dls, ics, max_cluster, res_in_brief, output)

    else:

        # categorical(data, exclus_info['attributes'], exclus_info['infor'], exclus_info['prior'], exclus_info['res_in_brief'], output)
        categorical1(data, exclus_info['attributes'], exclus_info['infor'], exclus_info['prior'],
                    exclus_info['res_in_brief'], output)



