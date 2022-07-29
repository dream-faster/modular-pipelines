from datasets import load_dataset

from data.dataloader import transform_dataset

from library.evaluation import calibration_metrics, classification_metrics
from library.examples.hate_speech import (  # preprocess_config,
    ensemble_pipeline, ensemble_pipeline_hf, ensemble_pipeline_hf_statistic,
    huggingface_baseline, nlp_sklearn, nlp_sklearn_autocorrect,
    nlp_sklearn_simple, text_statistics_pipeline)
from run import run
# from library.examples.hate_speech_multi_hf import ensemble_hf_multi_transformer

from type import PreprocessConfig, RunConfig

preprocess_config = PreprocessConfig(
    train_size=-1,
    val_size=-1,
    test_size=-1,
    input_col="text",
    label_col="label",
)


hate_speech_data = transform_dataset(
    load_dataset("tweet_eval", "hate"), preprocess_config
)

train_dataset, test_dataset = hate_speech_data
run_name = "hf-distilbert-roberta"

run_configs = [
    RunConfig(
        run_name=run_name,
        dataset=train_dataset,
        train=True,
        remote_logging=False,
        save_remote=False,
    ),

    RunConfig(
        run_name=run_name,
        dataset=test_dataset,
        train=False,
        save_remote=False,
        remote_logging=True,
    ),
]

for pipeline in [
    # ensemble_pipeline_hf,
    # huggingface_baseline,
    # nlp_sklearn,
    # nlp_sklearn_simple,
    # nlp_sklearn_autocorrect,
    # text_statistics_pipeline,
    # ensemble_pipeline,
    # ensemble_hf_multi_transformer
]:
    run(
        pipeline,
        preprocess_config,
        project_id="hate-speech-detection",
        run_configs=run_configs,
        metrics=classification_metrics + calibration_metrics,
    )
