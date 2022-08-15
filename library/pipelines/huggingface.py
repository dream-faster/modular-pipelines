def create_nlp_huggingface_pipeline(autocorrect: bool) -> Pipeline:
    return Pipeline(
        "hf_autocorrect" if autocorrect else "hf",
        input_data,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                HuggingfaceModel("hf-model", huggingface_config),
            ]
        ),
    )
