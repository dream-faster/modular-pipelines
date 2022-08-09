from typing import Callable, List, Optional, Tuple, Union


from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
)
from type import HuggingfaceConfig, LoadOrigin

from utils.printing import PrintFormats, multi_line_print


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from configs.constants import Const
import torch

device = 0 if torch.cuda.is_available() else -1


def _safe_load_training_pipeline(
    module: str, config: HuggingfaceConfig
) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizerBase]]:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            module, num_labels=config.num_classes
        )
        tokenizer = AutoTokenizer.from_pretrained(module)

        multi_line_print(
            f"Training: Model loaded {config.task_type.value}: {PrintFormats.BOLD}{module.__class__.__name__ if isinstance(module, PreTrainedModel) else module}{PrintFormats.END}",
            level=1,
        )
    except:
        multi_line_print(
            f"Training: Couldn't load {module} model or tokenizer. Skipping.", level=1
        )
        model, tokenizer = None, None

    return model, tokenizer


def _safe_load_inference_pipeline(
    module: Union[str, PreTrainedModel],
    config: HuggingfaceConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Optional[Callable]:
    try:
        if tokenizer is not None:
            loaded_pipeline = pipeline(
                task=config.task_type.value,
                model=module,
                tokenizer=tokenizer,
                device=device,
            )
        else:
            loaded_pipeline = pipeline(
                task=config.task_type.value, model=module, device=device
            )

        multi_line_print(
            f"Inference: Pipeline loaded {config.task_type.value}: {PrintFormats.BOLD}{module.__class__.__name__ if isinstance(module, PreTrainedModel) else module}{PrintFormats.END}",
            level=1,
        )

    except:
        multi_line_print(
            f"Inference: Couldn't load {module} pipeline. Skipping.", level=1
        )

        loaded_pipeline = None

    return loaded_pipeline


def safe_load(
    train: bool,
    module: Union[str, PreTrainedModel],
    config: HuggingfaceConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Tuple[
    Optional[Union[Callable, PreTrainedModel]], Optional[PreTrainedTokenizerBase]
]:

    if train and isinstance(module, str):
        return _safe_load_training_pipeline(module, config)
    else:
        return _safe_load_inference_pipeline(module, config, tokenizer), None


def determine_load_order(
    config: HuggingfaceConfig, paths: dict
) -> List[Tuple[LoadOrigin, str]]:
    if (
        hasattr(config, "preferred_load_origin")
        and config.preferred_load_origin is not None
    ):
        load_order = [
            (
                config.preferred_load_origin,
                paths[config.preferred_load_origin],
            )
        ]
    else:
        load_order = paths.items()

    return load_order


def get_paths(config: HuggingfaceConfig, parent_path: str, id: str) -> dict:
    return {
        LoadOrigin.local: f"{Const.output_pipelines_path}/{parent_path}/{id}",
        LoadOrigin.remote: f"{config.user_name}/{parent_path}-{id}"
        if not hasattr(config, "remote_name_override")
        or config.remote_name_override is None
        else config.remote_name_override,
        LoadOrigin.pretrained: config.pretrained_model,
    }
