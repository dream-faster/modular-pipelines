from blocks.pipeline import Pipeline
from blocks.base import DataSource
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.models.huggingface import HuggingfaceModel
from utils.list import remove_none
from type import HuggingfaceConfig


def create_nlp_huggingface_pipeline(
    input: DataSource, config: HuggingfaceConfig, autocorrect: bool
) -> Pipeline:
    return Pipeline(
        "hf_autocorrect" if autocorrect else "hf",
        input,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                HuggingfaceModel("hf-model", config),
            ]
        ),
    )
