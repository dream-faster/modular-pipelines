import json

def dump_json(data, path:str):
    with open(path, 'w') as f:
        json.dump(data, f)