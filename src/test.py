# import numpy as np
# import pandas as pd
#
# from distributions import categorical, kde
#
#
# c_data = np.array([[0.1],[0.2],[0.3],[0.4],[0.1],[0.1]])
#
# cluster_label = 0
#
# a_data = np.array([[0.1],[0.2],[0.3],[0.4],[0.1],[0.1],[0.2],[0.3],[0.5],[0.6],[0.7]])
#
# att_name = 'test'
# kernal = 'gaussian'
#
# q_c1 = np.percentile(c_data, 25)
# q_c3 = np.percentile(c_data, 75)
# iqr_c = q_c3 - q_c1
# q_a1 = np.percentile(a_data, 25)
# q_a3 = np.percentile(a_data, 75)
# iqr_a = q_a3 - q_a1
# min_c = min(np.std(c_data), iqr_c/1.34+0.00001)
# min_a = min(np.std(a_data), iqr_a/1.34+0.00001)
# bandwidth_c = 0.9 * min_c * c_data.shape[0] ** (-0.2)
# bandwidth_a = 0.9 * min_a * a_data.shape[0] ** (-0.2)
# # bandwidth_c = 0.5
# # bandwidth_a = 0.5
#
# plt = kde(c_data, a_data, cluster_label,att_name, kernal, bandwidth_c, bandwidth_a)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# 数据集
overall_data = np.array([[0.1], [0.2], [0.3], [0.4], [0.1], [0.1], [0.2], [0.3], [0.5], [0.6], [0.7]])
subset_data = np.array([[0.1], [0.2], [0.3], [0.4], [0.1], [0.1]])

# 拟合 KDE 模型
kde_overall = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(overall_data)
kde_subset = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(subset_data)

# 生成评估点
x_plot = np.linspace(0.1, 1, 1000)[:, np.newaxis]

# 计算密度值 (对数密度)
log_dens_overall = kde_overall.score_samples(x_plot)
log_dens_subset = kde_subset.score_samples(x_plot)

# 转换回密度值
density_overall = np.exp(log_dens_overall)
density_subset = np.exp(log_dens_subset)

# 找到 x=0.1 处的密度值
index_0_1 = np.argmin(np.abs(x_plot[:, 0] - 0.1))
density_at_0_1_overall = density_overall[index_0_1]
density_at_0_1_subset = density_subset[index_0_1]

# 计算局部缩放因子
local_scaling_factor = density_at_0_1_overall / density_at_0_1_subset

# 对整个密度曲线进行局部缩放
scaled_density_subset = density_subset * local_scaling_factor

# 绘制 KDE 曲线
plt.plot(x_plot[:, 0], density_overall, label='Overall Data', color='blue')
plt.plot(x_plot[:, 0], scaled_density_subset, label='Subset Data (Scaled)', color='red')

# 填充 KDE 曲线
plt.fill_between(x_plot[:, 0], density_overall, alpha=0.3, color='blue')
plt.fill_between(x_plot[:, 0], scaled_density_subset, alpha=0.3, color='red')

# 图例和显示
plt.legend()
plt.show()

