import json


def dump_json(data, path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def dump_str(data, path: str):
    with open(path, "w") as f:
        f.write(data)
