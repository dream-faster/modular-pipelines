from blocks.pipeline import Pipeline
from blocks.models.huggingface import HuggingfaceModel

from blocks.models.sklearn import SKLearnModel
from type import PreprocessConfig, HuggingfaceConfig, SKLearnConfig
from blocks.pipeline import Pipeline
from blocks.transformations import Lemmatizer, SpacyTokenizer
from blocks.data import DataSource, StrConcat, VectorConcat
from blocks.transformations.predicitions_to_text import PredictionsToText
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.transformations.sklearn import SKLearnTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
)


nb = MultinomialNB()
lg = LogisticRegression()
gb = GradientBoostingClassifier(n_estimators=100, max_depth=20, random_state=0)

sklearn_config = SKLearnConfig(
    force_fit=False,
    classifier=VotingClassifier(
        estimators=[("nb", nb), ("lg", lg), ("gb", gb)], voting="soft"
    ),
    one_vs_rest=False,
)

input_data = DataSource("input")


sklearn_model_seq = [
    SpacyTokenizer(),
    Lemmatizer(),
    SKLearnTransformation(
        TfidfVectorizer(
            max_features=100000,
            ngram_range=(1, 3),
        )
    ),
    SKLearnModel("model1", sklearn_config),
    PredictionsToText(),
]

nlp_sklearn = Pipeline(
    "nlp_sklearn",
    input_data,
    sklearn_model_seq,
)


def simple_sklearn() -> Pipeline:
    return nlp_sklearn
