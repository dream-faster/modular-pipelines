from .base import Transformation
from utils.spacy import get_spacy
from typing import List
from type import DataType


class SpacyTokenizer(Transformation):

    inputTypes = [DataType.List, DataType.Series]
    outputType = DataType.List

    def load(self, pipeline_id: str, execution_order: int) -> int:
        self.nlp = get_spacy()
        return super().load(pipeline_id, execution_order)

    def predict(self, dataset: List[str]) -> List:
        return [self.nlp(text) for text in dataset]
