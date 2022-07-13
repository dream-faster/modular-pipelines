from model.base import BaseModel
from typing import List, Any
import pandas as pd
from type import BaseConfig
from configs.constants import Const


class StatisticAugmenter(BaseModel):
    def __init__(self, num_synonyms: int):
        self.config = BaseConfig(force_fit=False)

    def preload(self):
        pass

    def fit(self, train_dataset: pd.DataFrame):
        pass

    def predict(self, test_dataset: pd.DataFrame) -> List[str]:
        test_dataset['statistic'] = test_dataset[Const.input_col].apply(lambda x: self.get_statistic(x))
        return get_statistic(test_dataset[Const.input_col], self.num_synonyms)

    def is_fitted(self) -> bool:
        return True

def num_words(sentence: str) -> int:
    return len(nltk.word_tokenize(sentence))

def get_statistic(sentence: str) -> List[str]:
    
    
