from data.dataloader import load_data
from run import run

from library.examples.hate_speech import (
    nlp_huggingface,
    nlp_huggingface_autocorrect,
    nlp_sklearn_autocorrect,
    preprocess_config,
)

hate_speech_data = load_data("data/original", preprocess_config)

for pipeline in [nlp_huggingface, nlp_huggingface_autocorrect, nlp_sklearn_autocorrect]:
    run(pipeline, hate_speech_data)
