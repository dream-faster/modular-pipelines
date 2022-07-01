from dataclasses import dataclass


@dataclass
class HuggingfaceConfig:
    user_name: str
    repo_name: str
    push_to_hub: bool = False


@dataclass
class GlobalPreprocessConfig:
    train_size: int
    test_size: int
    data_from_huggingface: bool


huggingface_config = HuggingfaceConfig(
    user_name="semy",
    repo_name="finetuning-sentiment-model-sst",
    push_to_hub=True
)

global_preprocess_config = GlobalPreprocessConfig(
    train_size=-1,
    test_size=5,
    data_from_huggingface=False,
)
