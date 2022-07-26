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
from utils.balance_labels import balance_labels

preprocess_config = PreprocessConfig(
    train_size=2000,
    val_size=2000,
    test_size=2000,
    input_col="text",
    label_col="label",
)

hate_speech_data = load_data("data/original", preprocess_config)


train_data, test_data = hate_speech_data

hate_speech_data = (balance_labels(train_data), balance_labels(test_data))

for pipeline in [
    # ensemble_pipeline_hf,
    # huggingface_baseline,
    # nlp_sklearn,
    nlp_sklearn_simple,
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
