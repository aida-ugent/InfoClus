import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import io
import anndata as ad

sys.path.insert(1, '../../src/')
from caching import from_cache
from pypdf import PdfReader, PdfWriter

with open('C:/Users/Administrator/OneDrive - UGent/Documents/Data/ExClus/german_socio_eco/german_socio_eco_original.csv', mode='r') as file:
    content = file.read().replace(',', '.')  # 将逗号替换为小数点
data = io.StringIO(content)
df = pd.read_csv(data, sep=';')

points = df['Coordinates'].str.split('. ', expand=True).astype(float).values
exclus_info = from_cache('german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0')
categories = exclus_info["clustering"]
names = df['Area Name'].values # 点的名称

category_0_idx = np.where(categories == 0)
category_1_idx = np.where(categories == 1)
category_2_idx = np.where(categories == 2)

size = 7
plt.figure()
plt.scatter(points[categories == 0, 1], points[categories == 0, 0], color='red', label='Cluster 0', s=size)
plt.scatter(points[categories == 1, 1], points[categories == 1, 0], color='#66c2a5', alpha=0.4, label='Cluster 1', s=size)
plt.scatter(points[categories == 2, 1], points[categories == 2, 0], color='blue', alpha=0.2, label='Cluster 2', s=size)

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
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

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize = 'large', markerscale = 2)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

textsize = 12
index = np.where(names == 'Brandenburg an der Havel')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Cottbus')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Vogtlandkreis')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Stralsund')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Uecker-Randow')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Wesermarsch')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Pirmasens')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Heilbronn')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='left')
index = np.where(names == 'Garmisch-Partenkirchen')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
# index = np.where(names == 'Berchtesgadener Land')[0][0]
# plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
plt.savefig('german_spatial_c0.pdf', bbox_inches='tight', pad_inches=0)

####################################################### cluster 2 urban #####################################################
plt.figure()
size = 7
plt.scatter(points[categories == 0, 1], points[categories == 0, 0], color='red',alpha=0.2, label='Cluster 0', s=size)
plt.scatter(points[categories == 1, 1], points[categories == 1, 0], color='#66c2a5', alpha=0.4, label='Cluster 1', s=size)
plt.scatter(points[categories == 2, 1], points[categories == 2, 0], color='blue', label='Cluster 2',alpha=0.8, s=size)

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
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

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize = 'large', markerscale = 2)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

textsize = 12
index = np.where(names == 'Hamburg')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Hannover')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Goettingen')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Osnabrueck')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Freiburg im Breisgau')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Passau')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Regensburg')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Tuebingen')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Trier')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Berlin. Stadtstaat')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Leverkusen')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
index = np.where(names == 'Gelsenkirchen')[0][0]
plt.text( points[index, 1],points[index, 0], names[index], fontsize=textsize, ha='center')
plt.savefig('german_spatial_c2.pdf', bbox_inches='tight', pad_inches=0)


##################################### clustering painting ###################################
adata = ad.read_h5ad(f'C:/Users/Administrator/trace/data/german_socio_eco/german_socio_eco.h5ad')
embedding = adata.obsm.get('tSNE_1')

exclus_info = from_cache('german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0')
clustering = exclus_info["clustering"]

plt.figure()
unique_classes = np.unique(clustering)

colors = ['red', 'green', 'blue']
for cls in unique_classes:
    indices = np.where(clustering == cls)
    plt.scatter(embedding[indices, 0], embedding[indices, 1], label=f'Cluster {cls}', alpha=0.7, color = colors[cls], s=7)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(unique_classes), fontsize = 'large', markerscale=2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().xaxis.set_ticks([])
plt.gca().yaxis.set_ticks([])

plt.savefig('intermediate_fig_clustering.pdf', bbox_inches='tight', pad_inches=0)


############################## top attributes ###############################################

writer = PdfWriter()
reader = PdfReader('german_socio_eco-tSNE_1-single-50-1.5-2-5-0-0.pdf')
page = reader.pages[0]
writer.add_page(page)
page = reader.pages[2]
writer.add_page(page)
page = reader.pages[5]
writer.add_page(page)
with open('german_top_attributes.pdf', 'wb') as output_pdf:
    writer.write(output_pdf)

################################# combine ######################################################

writer = PdfWriter()
pdf_files = ['german_spatial_c0.pdf', 'german_spatial_c2.pdf', 'intermediate_fig_clustering.pdf','german_top_attributes.pdf']

reader = PdfReader('intermediate_fig_clustering.pdf')
page = reader.pages[0]
writer.add_page(page)

reader = PdfReader('german_spatial_c0.pdf')
page = reader.pages[0]
writer.add_page(page)

reader = PdfReader('german_spatial_c2.pdf')
page = reader.pages[0]
writer.add_page(page)

reader = PdfReader('german_top_attributes.pdf')
page = reader.pages[0]
writer.add_page(page)
page = reader.pages[1]
writer.add_page(page)
page = reader.pages[2]
writer.add_page(page)

with open('German_Binder.pdf', 'wb') as output_pdf:
    writer.write(output_pdf)

print("PDF done!")
