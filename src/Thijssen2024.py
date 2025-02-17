import numpy as np
import pandas as pd
from infoclus import InfoClus
import matplotlib.pyplot as plt
import seaborn as sns

RADIUS = 1

def ranking_variance(v: np.ndarray, data: np.ndarray) -> np.ndarray:
    sum_pd_v = np.sum(v, axis=0)
    count_D = len(data)
    GV_v = np.sum((v - sum_pd_v / count_D) ** 2, axis=0) / count_D
    LV_v = np.var(v, axis=0)
    denominator = np.sum(LV_v/GV_v)
    s_variance = (LV_v/GV_v)/denominator
    return s_variance

def ranking_variance_fixed(v: np.ndarray, data: np.ndarray) -> np.ndarray:
    GV_v = np.var(data, axis=0)
    LV_v = np.var(v, axis=0)
    denominator = np.sum(LV_v/GV_v)
    s_variance_fixed = (LV_v/GV_v)/denominator
    return s_variance_fixed

def ranking_value(v: np.ndarray, data: np.ndarray) -> np.ndarray:
    LA_v = np.mean(v, axis=0)
    GA_data = np.mean(data, axis=0)
    GR_data = np.max(data, axis=0) - np.min(data, axis=0)
    denominator = np.sum(np.abs(LA_v - GA_data) / GR_data, axis=0)
    s_value = (LA_v-GA_data)/denominator
    return s_value

def get_lowest_value_index(s: np.ndarray) -> int:
    return np.argmin(s)

def get_highest_value_index(s: np.ndarray) -> int:
    return np.argmax(s)

def get_neighbor_mask(point: np.ndarray, data_ld: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(data_ld - point, axis=1)
    mask = distances <= RADIUS
    return mask

dataset_name = 'cytometry_2500'
embedding_name = 'tsne'
infoclus = InfoClus(dataset_name=dataset_name, main_emb=embedding_name)
data_hd = infoclus.data_scaled.values
data_ld = infoclus.all_embeddings[embedding_name]

count_instances = len(data_hd)
top_attributes_variance = []
top_attributes_variance_fixed = []
top_attributes_values = []
for i in range(count_instances):
    mask = get_neighbor_mask(data_ld[i], data_ld)
    neighbors = data_hd[mask]

    s_variance = ranking_variance(neighbors, data_hd)
    top_attributes_variance.append(get_lowest_value_index(s_variance))

    s_variance_fixed = ranking_variance_fixed(neighbors, data_hd)
    top_attributes_variance_fixed.append(get_lowest_value_index(s_variance_fixed))

    s_value = ranking_value(neighbors, data_hd)
    top_attributes_values.append(get_highest_value_index(s_value))

df = pd.DataFrame(data_ld, columns=['tsne_dim1', 'tsne_dim2'])
df['top_attribute_variance'] = top_attributes_variance_fixed
df['top_attribute_values'] = top_attributes_values

df.to_csv('output.csv', index=False, encoding='utf-8')
print(infoclus.data_raw.columns)

unique_classes = np.unique(top_attributes_variance_fixed)
num_classes = len(unique_classes)
colors = sns.color_palette("colorblind", num_classes)  # HUSL generates distinguishable colors
plt.figure(figsize=(8, 6))
for i, cls in enumerate(unique_classes):
    # Select points corresponding to the current class
    class_points = data_ld[top_attributes_variance_fixed == cls]
    lable = infoclus.data_raw.columns[cls]
    plt.scatter(class_points[:, 0], class_points[:, 1],
                color=colors[i], label=lable, s=15)
plt.legend()
plt.title("Cytometry 2500 colored by variance_fixed - Thijssen2024")
plt.savefig("Cytometry 2500 colored by variance_fixed - Thijssen2024")
plt.show()

unique_classes = np.unique(top_attributes_values)
num_classes = len(unique_classes)
colors = sns.color_palette("colorblind", num_classes)  # HUSL generates distinguishable colors
plt.figure(figsize=(8, 6))
for i, cls in enumerate(unique_classes):
    # Select points corresponding to the current class
    class_points = data_ld[top_attributes_values == cls]
    lable = infoclus.data_raw.columns[cls]
    plt.scatter(class_points[:, 0], class_points[:, 1],
                color=colors[i], label=lable, s=15)
plt.legend()
plt.title("Cytometry 2500 colored by values - Thijssen2024")
plt.savefig("Cytometry 2500 colored by values - Thijssen2024")
plt.show()


print('hello world')