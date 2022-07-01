from model.huggingface_classifier.base import HuggingfaceClassifyModel
from data.dataloader import load_data
from config import global_preprocess_config, huggingface_config


def run_pipeline():
    train_dataset, val_dataset, test_dataset = load_data(from_huggingface=False)
    train_dataset = train_dataset[: global_preprocess_config.train_size]
    test_dataset = test_dataset[: global_preprocess_config.test_size]

    model = HuggingfaceClassifyModel(
        data=(train_dataset, val_dataset, test_dataset), config=huggingface_config
    )

    model.fit()
    model.predict()


if __name__ == "__main__":
    run_pipeline()
