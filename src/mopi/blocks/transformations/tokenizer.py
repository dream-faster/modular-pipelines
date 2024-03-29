from typing import List

from mopi.type import DataType
from mopi.utils.spacy import get_spacy

from .base import Transformation


class SpacyTokenizer(Transformation):

    inputTypes = [DataType.List, DataType.Series]
    outputType = DataType.List

    def load(self) -> None:
        self.model = get_spacy()

    def predict(self, dataset: List[str]) -> List:
        return [self.model(text, disable=["parser", "tagger"]) for text in dataset]
