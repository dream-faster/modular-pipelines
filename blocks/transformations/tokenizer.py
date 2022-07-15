from .base import Transformation
from utils.spacy import get_spacy
from typing import List
from type import DataType


class SpacyTokenizer(Transformation):

    inputTypes = DataType.List_str
    outputType = DataType.List_str

    def load_remote(self):
        self.nlp = get_spacy()

    def predict(self, dataset: List[str]) -> List:
        return [self.nlp(text) for text in dataset]
