from huggingface_hub import HfApi


def initialize_huggingface(repo_id:str) -> HfApi:
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=True)

    return api
