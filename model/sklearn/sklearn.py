import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.base import ClassifierMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from model.base import Model
from type import SKLearnConfig, Label, Probabilities
from .preprocess import create_preprocess
import spacy
import swifter
from configs.constants import Const
from utils.spacy import get_spacy


class SKLearnModel(Model):

    config: SKLearnConfig

    def __init__(self, id: str, config: SKLearnConfig):
        self.id = id
        self.config = config

    def preload(self):
        self.nlp = get_spacy()
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.preprocess = create_preprocess(self.nlp)

    def fit(self, dataset: pd.DataFrame) -> None:

        X_train = dataset[Const.input_col].swifter.apply(self.preprocess)
        y_train = dataset[Const.label_col]

        self.pipeline = ImbPipeline(
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

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset[Const.input_col].swifter.apply(self.preprocess)
        predictions = self.pipeline.predict(dataset)
        probabilities = [tuple(row) for row in self.pipeline.predict_proba(dataset)]

        return pd.DataFrame(
            {Const.preds_col: predictions, Const.probs_col: probabilities}
        )

    def is_fitted(self) -> bool:
        return hasattr(self, "pipeline")


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
