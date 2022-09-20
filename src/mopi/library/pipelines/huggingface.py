from mopi.blocks.pipeline import Pipeline
from mopi.blocks.base import DataSource
from mopi.blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from mopi.blocks.models.huggingface import HuggingfaceModel
from mopi.utils.list import remove_none
from mopi.type import HuggingfaceConfig


def create_nlp_huggingface_pipeline(
    title: str, input: DataSource, config: HuggingfaceConfig, autocorrect: bool
) -> Pipeline:
    return Pipeline(
        title,
        input,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                HuggingfaceModel(
                    "hf-model", config, dict_lookup={"LABEL_0": 0, "LABEL_1": 1}
                ),
            ]
        ),
    )
