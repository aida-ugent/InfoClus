import sys
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../src/')
from caching import from_cache
from pypdf import PdfReader, PdfWriter

# --------------------------------Exclus clustering ploting------------------------------
adata = ad.read_h5ad(f'C:/Users/Administrator/trace/data/cytometry_2500/cytometry_2500.h5ad')
embedding = adata.obsm.get('tSNE_1')

exclus_info = from_cache('cytometry_2500-tSNE_1-single-700-1.5-2-10-0-0')
clustering = exclus_info["clustering"]

plt.figure(figsize=(8, 6))
unique_classes = np.unique(clustering)

for cls in unique_classes:
    indices = np.where(clustering == cls)
    plt.scatter(embedding[indices, 0], embedding[indices, 1], label=f'Cluster {cls}')

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
reader = PdfReader('cytometry_2500-tSNE_1-single-700-1.5-2-10-0-0.pdf')
page = reader.pages[0]
writer.add_page(page)
page = reader.pages[2]
writer.add_page(page)
page = reader.pages[7]
writer.add_page(page)
page = reader.pages[12]
writer.add_page(page)
page = reader.pages[16]
writer.add_page(page)
with open('cytometry_top_attributes.pdf', 'wb') as output_pdf:
    writer.write(output_pdf)

################################# combine ######################################################

writer = PdfWriter()
pdf_files = ['cytometry_fig1_manualGating.pdf', 'intermediate_fig_labelled_tSNE.pdf', 'intermediate_fig_clustering.pdf','cytometry_top_attributes.pdf']

##### binder1 ##############
reader = PdfReader('cytometry_fig1_manualGating.pdf')
page = reader.pages[0]
writer.add_page(page)

reader = PdfReader('intermediate_fig_labelled_tSNE.pdf')
page = reader.pages[0]
writer.add_page(page)

with open('cytometry_Binder_1.pdf', 'wb') as output_pdf:
    writer.write(output_pdf)

##### binder2 ##############
writer = PdfWriter()
reader = PdfReader('intermediate_fig_clustering.pdf')
page = reader.pages[0]
writer.add_page(page)

reader = PdfReader('cytometry_top_attributes.pdf')
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

with open('cytometry_Binder_2.pdf', 'wb') as output_pdf:
    writer.write(output_pdf)

print("PDF done!")


