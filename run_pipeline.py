from model.huggingface_classifier.base import HuggingfaceClassifyModel
from data.dataloader import load_data, shorten_datasets
from config import (
    global_preprocess_config,
    huggingface_config,
    BaseConfig,
    GlobalPreprocessConfig,
)


def run_pipeline(
    preprocess_config: GlobalPreprocessConfig, model_configs: dict[str, BaseConfig]
):
    data = load_data(from_huggingface=False)
    train_dataset, val_dataset, test_dataset = shorten_datasets(data, preprocess_config)

    model = HuggingfaceClassifyModel(config=model_configs["huggingface_classifier"])

    model.fit(train_dataset, val_dataset)
    model.predict(test_dataset)


if __name__ == "__main__":
    run_pipeline(
        global_preprocess_config, {"huggingface_classifier": huggingface_config}
    )
