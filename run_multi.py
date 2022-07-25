from data.dataloader import load_data
from run import run

from library.examples.hate_speech import (
    create_nlp_sklearn_pipeline,
    create_nlp_huggingface_pipeline,
    hate_speech_detection_pipeline,
    preprocess_config,
)

hate_speech_data = load_data("data/original", preprocess_config)

for pipeline_caller in [
    create_nlp_sklearn_pipeline,
    create_nlp_huggingface_pipeline,
    hate_speech_detection_pipeline,
]:
    pipeline = pipeline_caller()
    run(pipeline, hate_speech_data)
