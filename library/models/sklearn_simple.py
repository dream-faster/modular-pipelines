from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from type import SKLearnConfig

sklearn_config_simple_nb = SKLearnConfig(
    preferred_load_origin=None,
    frozen=False,
    save=True,
    classifier=MultinomialNB(),
    one_vs_rest=False,
    save_remote=False,
    calibrate=False,
)

sklearn_config_simple_lr = SKLearnConfig(
    preferred_load_origin=None,
    frozen=False,
    save=True,
    classifier=LogisticRegression(),
    one_vs_rest=False,
    save_remote=False,
    calibrate=False,
)
