import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import io
import anndata as ad

sys.path.insert(1, '../../src/')
from caching import from_cache
from pypdf import PdfReader, PdfWriter


##################################### clustering painting ###################################
adata = ad.read_h5ad(f'C:/Users/Administrator/trace/data/mushroom_binary/mushroom_binary.h5ad')
embedding = adata.obsm.get('tSNE_1')

exclus_info = from_cache('mushroom_binary-tSNE_1-single-800-1.5-2-30-0-0')
clustering = exclus_info["clustering"]

plt.figure(figsize=(8, 6))
unique_classes = np.unique(clustering)

for cls in unique_classes:
    indices = np.where(clustering == cls)
    plt.scatter(embedding[indices, 0], embedding[indices, 1], label=f'Cluster {cls}', s=3)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 0),
    ncol=len(unique_classes),
    markerscale=4,
    fontsize=15,  # 增大字体
    handletextpad=0.3,  # 缩小图例标记与文字的距离
    columnspacing=0.2   # 缩小列之间的间距
)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().xaxis.set_ticks([])
plt.gca().yaxis.set_ticks([])

plt.savefig('intermediate_fig_clustering.pdf')

############################## top attributes ###############################################

writer = PdfWriter()
reader = PdfReader('mushroom_binary-tSNE_1-single-800-1.5-2-30-0-0.pdf')
page = reader.pages[0]
writer.add_page(page)
page = reader.pages[2]
writer.add_page(page)
page = reader.pages[7]
writer.add_page(page)
page = reader.pages[13]
writer.add_page(page)
page = reader.pages[15]
writer.add_page(page)
with open('mushroom_top_attributes.pdf', 'wb') as output_pdf:
    writer.write(output_pdf)

################################# combine ######################################################

writer = PdfWriter()
pdf_files = ['intermediate_fig_clustering.pdf', 'mushroom_top_attributes.pdf']

reader = PdfReader('intermediate_fig_clustering.pdf')
page = reader.pages[0]
writer.add_page(page)

reader = PdfReader('mushroom_top_attributes.pdf')
page = reader.pages[0]
writer.add_page(page)
page = reader.pages[1]
writer.add_page(page)
page = reader.pages[2]
writer.add_page(page)
page = reader.pages[3]
writer.add_page(page)
page = reader.pages[4]
writer.add_page(page)

with open('mushroom_Binder.pdf', 'wb') as output_pdf:
    writer.write(output_pdf)

print("PDF done!")
