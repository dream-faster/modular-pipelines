from cgi import test
from dataclasses import dataclass
from typing import Optional
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
from type import PreprocessConfig, RunConfig

preprocess_config = PreprocessConfig(
    train_size=100,
    val_size=100,
    test_size=100,
    input_col="text",
    label_col="label",
)


hate_speech_data = load_data("data/original", preprocess_config)

train_dataset, test_dataset = hate_speech_data
run_name = "nlp-ensemble"
run_configs = [
    RunConfig(
        run_name=run_name, dataset=train_dataset, train=True, remote_logging=False
    ),
    RunConfig(run_name=run_name, dataset=test_dataset, train=False),
]

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
        preprocess_config,
        project_id="hate-speech-detection",
        run_configs=run_configs,
    )
