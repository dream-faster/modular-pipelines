from sklearn.naive_bayes import MultinomialNB
from type import SKLearnConfig

sklearn_config_simple = SKLearnConfig(
    preferred_load_origin=None,
    frozen=False,
    save=True,
    classifier=MultinomialNB(),
    one_vs_rest=False,
    save_remote=False,
    calibrate=False,
)
