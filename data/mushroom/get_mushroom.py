import pandas as pd
from ucimlrepo import fetch_ucirepo

def parse_names_file(file_path):
    mapping_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                key, values = line.split(":")
                value_dict = {v: k for k, v in (item.split("=") for item in values.split(","))}  # 反转 key-value
                mapping_dict[key] = value_dict
    return mapping_dict

# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

df = pd.concat([X, y], axis=1)
# replace values in df by names.txt
names_mapping = parse_names_file("names.txt")
for column, mapping in names_mapping.items():
    if column in df.columns:  # 只有当列名匹配时才替换
        df[column] = df[column].replace(mapping)


df.to_csv("mushroom.csv", index=False)
print("CSV file saved as mushroom.csv")
