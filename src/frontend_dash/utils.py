import os
import pickle


def check_infoclus_object(dataset_name: str,embedding_name: str):
    script_a_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_a_dir, '..', '..', 'data', dataset_name)
    if_exists = os.path.exists(os.path.join(data_folder, f'{dataset_name}_{embedding_name}.pkl'))
    if if_exists:
        with open(os.path.join(data_folder, f'{dataset_name}_{embedding_name}.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        return None
