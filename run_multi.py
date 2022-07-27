from data.dataloader import load_data
from run import run


from library.examples.hate_speech import (
    huggingface_baseline,
    nlp_sklearn,
    nlp_sklearn_autocorrect,
    text_statistics_pipeline,
    ensemble_pipeline,
    ensemble_pipeline_hf,
    ensemble_pipeline_hf_statistic,
    nlp_sklearn_simple,
    # preprocess_config,
)
from type import PreprocessConfig

preprocess_config = PreprocessConfig(
    train_size=100,
    val_size=100,
    test_size=100,
    input_col="text",
    label_col="label",
)

hate_speech_data = load_data("data/original", preprocess_config)

for pipeline in [
    # ensemble_pipeline_hf,
    huggingface_baseline,
    # nlp_sklearn,
    # nlp_sklearn_simple,
    # nlp_sklearn_autocorrect,
    # text_statistics_pipeline,
    # ensemble_pipeline,
]:
    run(
        pipeline,
        hate_speech_data,
        preprocess_config,
        project_id="hate-speech-detection",
    )
