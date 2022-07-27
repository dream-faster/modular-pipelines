import os  
from dotenv import load_dotenv  
import warnings
from typing import Optional
    
def get_env(key:str)->Optional[str]:    
    try:
        load_dotenv()
        value = os.environ.get(key)
        return value
    except:
        warnings.warn(f"Couldn't load {key}")
        return None
    