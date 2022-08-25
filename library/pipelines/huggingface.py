from blocks.pipeline import Pipeline
from blocks.base import DataSource
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.models.huggingface import HuggingfaceModel
from utils.list import remove_none
from type import HuggingfaceConfig


def create_nlp_huggingface_pipeline(
    title: str, input: DataSource, config: HuggingfaceConfig, autocorrect: bool
) -> Pipeline:
    return Pipeline(
        title,
        input,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                HuggingfaceModel("hf-model", config, dict_lookup={"LABEL_0": 0, "LABEL_1": 1}),
            ]
        ),
    )
