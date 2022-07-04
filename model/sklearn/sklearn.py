import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.base import ClassifierMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from model.base import BaseModel
from type import SKLearnConfig, Label, Probabilities
from .preprocess import preprocess
import spacy
from typing import Tuple, List
import swifter


class SKLearnModel(BaseModel):

    config: SKLearnConfig

    def __init__(self, config: SKLearnConfig):
        self.config = config
        nlp = spacy.load("en_core_web_sm")
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def fit(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame) -> None:

        X_train = train_dataset["text"].swifter.apply(preprocess)
        y_train = train_dataset["label"]

        X_val = val_dataset["text"].swifter.apply(preprocess)
        y_val = val_dataset["label"]

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        min_df=2,
                        max_df=0.5,
                        stop_words=self.spacy_stopwords,
                        max_features=100000,
                        ngram_range=(1, 3),
                    ),
                ),
                # ('feature_selection', SelectPercentile(chi2, percentile=50)),
                ("sampling", RandomOverSampler()),
                ("clf", self.config.classifier),
            ],
            verbose=True,
        )

        self.pipeline.fit(X_train, y_train)
        f1 = f1_score(y_val, self.pipeline.predict(X_val), average="weighted")
        print(f"{type(self.config.classifier)} f1: {f1}")

    def predict(self, test_dataset: pd.DataFrame) -> List[Tuple[Label, Probabilities]]:
        prerocessed_dataset = test_dataset["text"].swifter.apply(preprocess)
        predictions = self.pipeline.predict(prerocessed_dataset)
        probabilities = self.pipeline.predict_proba(prerocessed_dataset)
        return list(zip(predictions, probabilities))


def create_classifier(
    classifier: ClassifierMixin, one_vs_rest: bool
) -> ClassifierMixin:
    if one_vs_rest:
        return classifier
    else:
        return OneVsRestClassifier(
            ("clf", classifier),
            n_jobs=4,
        )