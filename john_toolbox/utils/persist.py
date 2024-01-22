import json
import logging
import os
import pickle
import shutil

import yaml

logger = logging.getLogger(__name__)


def save_pickle(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    return obj


def remove_pickle(file_path):
    if os.path.exists(file_path):
        # removing the file using the os.remove() method
        os.remove(file_path)
    else:
        # file not found message
        print("File not found in the directory.")


def remove_folder(path):
    try:
        shutil.rmtree(path)
        print("Directory removed successfully")
    except OSError as o:
        print(f"Error, {o.strerror}: {path}")


def save_json(obj, path):
    with open(path, "w") as outfile:
        json.dump(obj, outfile, indent=4)


def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def save_yaml(obj, path):
    with open(path, "w") as file:
        yaml.dump(obj, file, Dumper=yaml.Dumper)


def load_yaml(path):
    with open(path, "r") as stream:
        obj = yaml.load(stream, Loader=yaml.SafeLoader)
    return obj


def pretty_json(obj: dict):
    return json.dumps(obj, sort_keys=False, indent=4, separators=(",", ": "))


def is_folder_exist(folder_path: str):
    return os.path.isdir(folder_path)


def list_dir(folder_path: str):
    return os.listdir(folder_path)


def create_local_folder(local_dirs: str):
    try:
        os.makedirs(local_dirs)
        logger.debug(f"local_dirs {local_dirs} created successfully")
    except FileExistsError:
        logger.debug("no need to create folders as it already exists.")
