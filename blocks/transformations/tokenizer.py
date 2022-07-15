from .base import Transformation
from utils.spacy import get_spacy
from typing import List


class SpacyTokenizer(Transformation):
    def load_remote(self):
        self.nlp = get_spacy()

    def predict(self, dataset: List[str]) -> List:
        return [self.nlp(text) for text in dataset]
