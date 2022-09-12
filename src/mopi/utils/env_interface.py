import os
import warnings
from typing import Optional

from dotenv import load_dotenv


def get_env(key: str) -> Optional[str]:
    try:
        load_dotenv()
        value = os.environ.get(key)
        return value
    except:
        warnings.warn(f"Couldn't load {key}")
        return None
