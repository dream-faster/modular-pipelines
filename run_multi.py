from data.dataloader import load_data
from run import run

from library.examples.hate_speech import (
    huggingface_baseline,
    nlp_sklearn,
    nlp_sklearn_autocorrect,
    text_statistics_pipeline,
    ensemble_pipeline,
    preprocess_config,
)

hate_speech_data = load_data("data/original", preprocess_config)

for pipeline in [
    # huggingface_baseline,
    nlp_sklearn,
    nlp_sklearn_autocorrect,
    text_statistics_pipeline,
    # ensemble_pipeline,
]:
    run(pipeline, hate_speech_data)
