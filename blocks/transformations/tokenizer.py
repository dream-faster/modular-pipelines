from typing import List

from type import DataType
from utils.spacy import get_spacy

from .base import Transformation


class SpacyTokenizer(Transformation):

    inputTypes = [DataType.List, DataType.Series]
    outputType = DataType.List

    def load(self) -> None:
        self.nlp = get_spacy()

    def predict(self, dataset: List[str]) -> List:
        return [self.nlp(text) for text in dataset]
