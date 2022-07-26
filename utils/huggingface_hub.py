from huggingface_hub import HfApi


def initialize_huggingface(token: str) -> None:
    api = HfApi()
    api.set_access_token(token)
