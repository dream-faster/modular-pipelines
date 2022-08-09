from typing import Callable, List, Optional, Tuple, Union

from type import HuggingfaceConfig, LoadOrigin

from utils.printing import PrintFormats, multi_line_print

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from configs.constants import Const
import torch

device = 0 if torch.cuda.is_available() else -1


def safe_load(
    module: Union[str, PreTrainedModel],
    config: HuggingfaceConfig,
) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            module, num_labels=config.num_classes
        )
        tokenizer = AutoTokenizer.from_pretrained(module)

        multi_line_print(
            f"Training: Model loaded {config.task_type.value}: {PrintFormats.BOLD}{module.__class__.__name__ if isinstance(module, PreTrainedModel) else module}{PrintFormats.END}",
            level=1,
        )
        return model, tokenizer

    except:
        multi_line_print(
            f"Training: Couldn't load {module} model or tokenizer. Skipping.", level=1
        )
        return None, None


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
