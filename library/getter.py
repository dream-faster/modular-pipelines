from .examples.simple_decoder import simple_decoder
from .examples.text_image_pipeline import text_image_pipeline
from configs.constants import ModelTypes

def get_full_pipeline(pipeline:str):
    assert pipeline in vars(ModelTypes).keys(), "No such pipeline exists."
    
    if pipeline == "simple":
        return simple_decoder()
    elif pipeline == "text_image":
        return text_image_pipeline()
    
    