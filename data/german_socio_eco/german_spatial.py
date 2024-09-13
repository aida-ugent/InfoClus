import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import io

sys.path.insert(1, '../../src/')
from caching import from_cache


with open('C:/Users/Administrator/OneDrive - UGent/Documents/Data/ExClus/german_socio_eco/german_socio_eco_original.csv', mode='r') as file:
    content = file.read().replace(',', '.')  # 将逗号替换为小数点
data = io.StringIO(content)
df = pd.read_csv(data, sep=';')

points = df['Coordinates'].str.split('. ', expand=True).astype(float).values
exclus_info = from_cache('german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0')
categories = exclus_info["clustering"]
names = df['Area Name'].values # 点的名称

# 获取每个类别的索引
category_0_idx = np.where(categories == 0)
category_1_idx = np.where(categories == 1)
category_2_idx = np.where(categories == 2)

size = 7
plt.scatter(points[categories == 0, 1], points[categories == 0, 0], color='red', label='Cluster 0', s=size)
plt.scatter(points[categories == 1, 1], points[categories == 1, 0], color='blue', alpha=0.4, label='Cluster 1', s=size)
plt.scatter(points[categories == 2, 1], points[categories == 2, 0], color='orange', alpha=0.4, label='Cluster 2', s=size)

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
# 在图像的中央添加背景文字
plt.text(
    (xlim[1] + xlim[0]) / 2,  # x 坐标为 x 轴范围的中点
    (ylim[1] + ylim[0]) / 2,  # y 坐标为 y 轴范围的中点
    'Germany',        # 要显示的文本
    color='black',            # 文本颜色
    fontsize=30,              # 文本大小
    ha='center',              # 水平对齐方式
    va='center',              # 垂直对齐方式
    alpha=0.1,                # 透明度
    weight='bold'             # 字体粗细
)


# 设置图例和标题
plt.legend(loc = 'best', fontsize='small')

plt.gca().set_aspect('equal', adjustable='box')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

# 隐藏坐标轴的刻度和标签
plt.xticks([])
plt.yticks([])


textsize = 8
# index = np.where(names == 'Wesermarsch')
# plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='right')
index = np.where(names == 'Brandenburg an der Havel')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='right')
index = np.where(names == 'Cottbus')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='right')
index = np.where(names == 'Vogtlandkreis')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='right')
index = np.where(names == 'Stralsund')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='right')
index = np.where(names == 'Uecker-Randow')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='right')
count = 0
for i in category_0_idx[0]:
    plt.text( points[i, 1],points[i, 0], names[i], fontsize=textsize, ha='right')
    count += 1
    if count > 5:
        break

# 显示图像
plt.savefig('german_spatial.pdf')