from huggingface_hub import HfApi


def initialize_huggingface(token: str) -> None:
    try:
        api = HfApi()
        api.set_access_token(token)
        print("API token set.")
    except:
        print("Token couldn't be set")
