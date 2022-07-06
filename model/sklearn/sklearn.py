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
from spacy.cli.download import download
from typing import Tuple, List, Any
import swifter
from configs.constants import Const


class SKLearnModel(BaseModel):

    config: SKLearnConfig

    def __init__(self, config: SKLearnConfig, train: bool, name: str):
        self.config = config
        self.train = train
        self.name = name

    def preload(self):
        try:
            spacy.load("en_core_web_lg")
        except:
            download("en_core_web_lg")
            spacy.load("en_core_web_lg")

        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def fit(self, train_dataset: pd.DataFrame) -> None:

        X_train = train_dataset[Const.input_col].astype(str).swifter.apply(preprocess)
        y_train = train_dataset[Const.label_col]

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

    def predict(self, test_dataset: List[Any]) -> List[Any]:
        prerocessed_dataset = [preprocess(line) for line in test_dataset]
        predictions = self.pipeline.predict(prerocessed_dataset)
        probabilities = self.pipeline.predict_proba(prerocessed_dataset)
        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return False


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
