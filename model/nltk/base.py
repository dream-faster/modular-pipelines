import nltk
from nltk.corpus import wordnet as wn
from model.base import BaseModel
from typing import List


class NLTKModel(BaseModel):
    def __init__(self):
        self.__download_libraries()

    def __download_libraries(self):
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    def fit(self, train_dataset, val_dataset):
        pass

    def infer(self, words: List[str])->List[List[str]]:
        return [
            [single_word for ss in wn.synsets(word) for single_word in ss.lemma_names()]
            for word in words
        ]
