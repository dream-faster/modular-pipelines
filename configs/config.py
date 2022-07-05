from type import HuggingfaceConfig, GlobalPreprocessConfig, SKLearnConfig

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)

huggingface_config = HuggingfaceConfig(
    pretrained_model="distilbert-base-uncased",
    epochs=2,
    user_name="itchingpixels",
    repo_name="finetuning-tweeteval-hate-speech",
    push_to_hub=True,
    num_classes=2,
    val_size=0.1,
    force_fit=False,
)

global_preprocess_config = GlobalPreprocessConfig(
    train_size=-1,
    val_size=-1,
    test_size=5,
    data_from_huggingface=False,
    input_name="text",
    label_name="label",
)

nb = MultinomialNB()
lg = LogisticRegression()
gb = GradientBoostingClassifier(n_estimators=100, max_depth=20, random_state=0)

# sklearn_config = SKLearnConfig(
#     classifier=VotingClassifier(
#         estimators=[("nb", nb), ("lg", lg), ("gb", gb)], voting="soft"
#     ),
#     one_vs_rest=False,
# )
sklearn_config = SKLearnConfig(classifier=nb, one_vs_rest=False, force_fit=False)
