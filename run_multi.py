from run import run

from library.examples.hate_speech import (
    nlp_huggingface,
    nlp_huggingface_autocorrect,
    nlp_sklearn_autocorrect,
)

for pipeline in [nlp_huggingface, nlp_huggingface_autocorrect, nlp_sklearn_autocorrect]:
    run(pipeline)
