import os, pickle, base64, io
import json
import re
from urllib import request, error, parse
import pandas as pd
import numpy as np
from infoclus import InfoClus
from caching import from_cache
from config import PROJECT_ROOT, DATA_FOLDER


def build_infoclus(dataset_name: str='german_socio_eco', emb_name='tsne', linkage='single', modify=True):

    cache_folder = os.path.join(PROJECT_ROOT, 'data', dataset_name, 'cache')
    file_path = os.path.join(cache_folder, emb_name +'_'+ linkage + '_' + 'modify_' + str(modify))

    if os.path.exists(file_path):
        infoclus = from_cache(file_path)
    else:
        infoclus = InfoClus(dataset_name=dataset_name, linkage=linkage, modify_hierarchical=modify)
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

def get_embeddings_keys(dataset_name):
    embeddings_keys = np.load(os.path.join(DATA_FOLDER, dataset_name, 'embeddings.npz')).files
    return embeddings_keys


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

def save_embedding_in_file(embedding, emb_name, dataset_name):
    embeddings_path = os.path.join(DATA_FOLDER, dataset_name, 'cache', 'embeddings.npz')
    embeddings = np.load(embeddings_path)
    embeddings[emb_name] = embedding
    np.savez(embeddings_path, **embeddings)


def get_labels_from_input(contents):
    if contents is None:
        return []

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df[df.columns[0]].values.tolist()


def create_paragraph_infoclus_explanation(dataset_store=None, infoclus_store=None):
    if isinstance(dataset_store, pd.DataFrame):
        df_data = dataset_store
    elif isinstance(dataset_store, str):
        decoded_dataset = deserialize_obj(dataset_store)
        df_data = decoded_dataset if isinstance(decoded_dataset, pd.DataFrame) else None
    else:
        df_data = None

    if isinstance(infoclus_store, dict):
        infoclus_dict = infoclus_store
    elif isinstance(infoclus_store, str):
        decoded_infoclus = deserialize_obj(infoclus_store)
        infoclus_dict = decoded_infoclus if isinstance(decoded_infoclus, dict) else None
    else:
        infoclus_dict = None

    if not isinstance(infoclus_dict, dict):
        return "Here is the explanation computed by InfoClus for each cluster."

    attributes_opt = infoclus_dict.get('attributes_opt')
    statistics = infoclus_dict.get('statistics')
    prior = infoclus_dict.get('prior')

    if attributes_opt is None or statistics is None or prior is None:
        return "Here is the explanation computed by InfoClus for each cluster."

    cluster_parts = []
    for cluster_idx, cluster_attrs in enumerate(attributes_opt):
        att_parts = []
        for attribute in cluster_attrs:
            if df_data is not None and isinstance(attribute, (int, np.integer)) and 0 <= int(attribute) < len(df_data.columns):
                att_name = str(df_data.columns[int(attribute)])
            else:
                att_name = str(attribute)

            stat_mean = float(statistics[cluster_idx][0][attribute])
            prior_mean = float(prior[0][attribute])
            mean_relation = "larger" if stat_mean > prior_mean else "smaller"
            att_parts.append(f"attribute {att_name} has mean value {mean_relation} than the prior mean")

        if att_parts:
            cluster_parts.append(f"for cluster {cluster_idx}, " + ", ".join(att_parts))

    if not cluster_parts:
        return "Here is the explanation computed by InfoClus for each cluster."

    return "Here is the explanation computed by InfoClus for each cluster, " + "; ".join(cluster_parts) + "."


def create_prompt(infoclus_store=None, dataset_store=None, task=None):
    if isinstance(dataset_store, pd.DataFrame):
        df_data = dataset_store
    elif isinstance(dataset_store, str):
        decoded_dataset = deserialize_obj(dataset_store)
        df_data = decoded_dataset if isinstance(decoded_dataset, pd.DataFrame) else None
    else:
        df_data = None

    if isinstance(infoclus_store, dict):
        infoclus_dict = infoclus_store
    elif isinstance(infoclus_store, str):
        decoded_infoclus = deserialize_obj(infoclus_store)
        infoclus_dict = decoded_infoclus if isinstance(decoded_infoclus, dict) else None
    else:
        infoclus_dict = None

    data_name = "dataset"
    if isinstance(infoclus_dict, dict) and infoclus_dict.get('data_name') is not None:
        data_name = str(infoclus_dict.get('data_name'))

    columns_text = "unknown columns"
    if isinstance(df_data, pd.DataFrame):
        columns_text = ", ".join([str(c) for c in df_data.columns])

    cluster_depiction = create_paragraph_infoclus_explanation(dataset_store=dataset_store, infoclus_store=infoclus_store)

    if task is None or str(task).strip() == "":
        task_text = (
            "Since the user has not specified a particular objective for the dataset, "
            "assume a commonly performed analysis task that aligns with the given data type "
            "and its attributes."
        )
    else:
        task_text = str(task)

    return (
        f"You are given a dataset called {data_name}, which contains columns {columns_text}. "
        "The data has already been divided into several clusters. Your task is to assign an "
        "appropriate label to each cluster.\n"
        "First, refer to the task description to understand the type or theme of labels you should use. "
        "Then, use the cluster description information to determine which label best fits each cluster.\n\n"
        f"Task: {task_text}\n\n"
        f"Cluster description information: {cluster_depiction}\n\n"
        "You must respond strictly using this format: cluster 0: {label0}, cluster 1: {label1}, … "
    )


def query_llm(provider, api_key, model, prompt, endpoint=None, timeout=60):
    provider = (provider or "").strip().lower()
    api_key = (api_key or "").strip()
    model = (model or "").strip()
    prompt = (prompt or "").strip()
    endpoint = (endpoint or "").strip()

    if not provider:
        return "Error: API provider is required."
    if not api_key:
        return "Error: API key is required."
    if not model:
        return "Error: Model name is required."
    if not prompt:
        return "Error: Prompt is empty. Please click Create prompt first."

    headers = {"Content-Type": "application/json"}
    url = endpoint
    payload = None

    if provider == "openai":
        url = url or "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
    elif provider == "anthropic":
        url = url or "https://api.anthropic.com/v1/messages"
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif provider == "google":
        base_url = url or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        sep = "&" if "?" in base_url else "?"
        url = f"{base_url}{sep}key={parse.quote(api_key)}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
    elif provider == "custom":
        if not url:
            return "Error: Custom provider requires API endpoint."
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        return f"Error: Unsupported provider '{provider}'."

    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)

        if provider in ("openai", "custom"):
            return data["choices"][0]["message"]["content"]
        if provider == "anthropic":
            content = data.get("content", [])
            if content and isinstance(content, list):
                return "".join([item.get("text", "") for item in content if isinstance(item, dict)])
            return json.dumps(data)
        if provider == "google":
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
            return json.dumps(data)

        return json.dumps(data)
    except error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        return f"HTTP Error {e.code}: {err_body}"
    except Exception as e:
        return f"Error while calling LLM: {e}"


def parse_llm_cluster_labels(response_text, expected_cluster_ids=None):
    text = (response_text or "").strip()
    if not text:
        return None, "LLM response is empty."

    # Expected format example: cluster 0: label0, cluster 1: label1
    pattern = re.compile(r"cluster\s*(\d+)\s*:\s*([^,;\n]+)", re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        return None, "LLM response format is invalid. Expected: cluster 0: {label0}, cluster 1: {label1}, ..."

    mapping = {}
    for cluster_id_str, label in matches:
        cid = int(cluster_id_str)
        label_clean = label.strip()
        if not label_clean:
            return None, f"Label for cluster {cid} is empty."
        mapping[cid] = label_clean

    if expected_cluster_ids is not None:
        expected_set = set(int(i) for i in expected_cluster_ids)
        parsed_set = set(mapping.keys())
        missing = sorted(expected_set - parsed_set)
        if missing:
            return None, f"Missing labels for clusters: {missing}."

    return mapping, None
