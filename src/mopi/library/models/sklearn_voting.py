from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from mopi.type import SKLearnConfig
from mopi.constants import Const

sklearn_config = SKLearnConfig(
    frozen=False,
    save=True,
    preferred_load_origin=None,
    classifier=VotingClassifier(
        estimators=[
            ("nb", MultinomialNB()),
            ("lg", LogisticRegression(random_state=Const.seed)),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100, max_depth=20, random_state=Const.seed
                ),
            ),
        ],
        voting="soft",
    ),
    one_vs_rest=False,
    save_remote=False,
    calibrate=False,
)
