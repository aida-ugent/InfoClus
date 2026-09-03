import os
import pickle


def from_cache(path_to_cache_file):
    if os.path.exists(path_to_cache_file):
        with open(path_to_cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None


def to_cache(path_to_cache_file, data):
    os.makedirs(os.path.dirname(path_to_cache_file), exist_ok=True)
    with open(path_to_cache_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
