import json
import pickle
import yaml


def save_pickle(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    return obj


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
