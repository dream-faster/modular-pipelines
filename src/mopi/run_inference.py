from mopi.type import Experiment
from mopi.blocks.pipeline import Pipeline

from mopi.type import (
    Experiment,
    DatasetSplit,
)
from mopi.data.dataloader import PandasDataLoader
from mopi.type import PreprocessConfig
import pandas as pd
from mopi.library.evaluation.classification import classification_metrics
from mopi.constants import Const

from typing import Tuple, List

from mopi.plugins import OutputAnalyserPlugin
from mopi.runner.runner import Runner
from mopi.type import Experiment, StagingConfig, StagingNames
from mopi.runner.utils import overwrite_preprocessing_configs_
from mopi.blocks.io import load_pipeline


def inference(
    experiment: Experiment,
    staging_config: StagingConfig,
) -> Tuple[Experiment, "Pipeline", "Store"]:

    overwrite_preprocessing_configs_(experiment.pipeline, staging_config)

    runner = Runner(
        experiment,
        plugins=[OutputAnalyserPlugin()],
    )

    store, pipeline, _ = runner.infer()

    return experiment, pipeline, store


def run_inference(pipeline: Pipeline, texts: List[str]) -> Tuple[int, float]:

    text_with_fake_labels = [[text, 0] for text in texts] + [["dummy_text", 1]]
    project_name = pipeline.run_context.project_name
    dataloader = PandasDataLoader(
        "",
        PreprocessConfig(
            train_size=-1,
            val_size=-1,
            test_size=-1,
            input_col="text",
            label_col="label",
        ),
        pd.DataFrame([["", ""]], columns=["input", "label"]),
        pd.DataFrame(text_with_fake_labels, columns=["input", "label"]),
    )

    experiment_for_inference = Experiment(
        project_name=project_name,
        run_name="inference",
        dataset_category=DatasetSplit.test,
        pipeline=pipeline,
        metrics=classification_metrics,
        train=False,
        global_dataloader=dataloader,
        save_remote=False,
        log_remote=False,
    )

    prod_config = StagingConfig(
        name=StagingNames.prod,
        save_remote=False,
        log_remote=True,
        limit_dataset_to=None,
    )

    successes = inference(
        experiment_for_inference,
        staging_config=prod_config,
    )

    _, _, store = successes
    results = store.get_all_predictions()[Const.final_output][: len(texts)]

    return results


if __name__ == "__main__":
    pipeline_name = "hf-distillbert"
    example_texts = ["This is an example text.", "Another example text."]
    pipeline = load_pipeline(pipeline_name)
    results = run_inference(pipeline, example_texts)

    for text, result in zip(example_texts, results):
        print(
            f"Results for '{text}' is {'hate speech ❌' if result == 1 else 'non-hate speech ✅'} ({result})"
        )
