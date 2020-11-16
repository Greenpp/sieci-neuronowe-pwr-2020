import pickle as pkl
from typing import Iterable


def save_to_pkl(f_name: str, dir: str, data: dict) -> None:
    with open(f'{dir}/{f_name}.pkl', 'wb') as f:
        pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)


def merge_results(files: Iterable, dir: str) -> dict:
    data = []
    for f_name in files:
        with open(f'{dir}/{f_name}.pkl', 'rb') as f:
            f_data = pkl.load(f)
            data.append(f_data)

    merged_data = data[0]
    for log in data[1:]:
        merged_data['results'].update(log['results'])

    return merged_data


def get_data(files: Iterable, dir: str) -> dict:
    data = []
    for f_name in files:
        with open(f'{dir}/{f_name}.pkl', 'rb') as f:
            f_data = pkl.load(f)
            data.append(f_data)

    return data