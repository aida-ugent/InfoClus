import os, pickle, base64, io
import pandas as pd
from infoclus import InfoClus
from caching import from_cache
from config import PROJECT_ROOT, DATA_FOLDER


def build_infoclus(dataset_name: str='german_socio_eco', emb_name='tsne', linkage='single', modify=True):

    cache_folder = os.path.join(PROJECT_ROOT, 'data', dataset_name, 'cache')
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    
    file_path = os.path.join(cache_folder, emb_name +'_'+ linkage + '_' + 'modify_' + str(modify))

    if os.path.exists(file_path):
        infoclus = from_cache(file_path)
    else:
        infoclus = InfoClus(dataset_name=dataset_name, emb_name=emb_name, linkage=linkage, modify_hierarchical=modify)
    return infoclus

def serialize_obj(obj):
    try:
        pickled = pickle.dumps(obj)
        encoded = base64.b64encode(pickled).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"Serialization failed: {e}")
        return None

def deserialize_obj(data: str):
    try:
        decoded = base64.b64decode(data.encode('utf-8'))
        obj = pickle.loads(decoded)
        return obj
    except Exception as e:
        print(f"Deserialization failed: {e}")
        return None

def get_datasets():
    folders = [f for f in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, f))]
    return folders


def save_dataset_in_folder(contents, filename, base_path=DATA_FOLDER):

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    folder_name = os.path.splitext(filename)[0]

    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'cache'), exist_ok=True)


    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        f.write(decoded)

    return file_path


def get_labels_from_input(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df[df.columns[0]].values.tolist()


