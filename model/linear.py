from configs.constants import Const
from .base import BaseModel
import pandas as pd
from type import BaseConfig
from typing import List, Any, Union
import numpy as np


class GlobalState:
    def __init__(self):
        self.state = dict()

    def get(self, key: str):
        return self.state[key]

    def set(self, key: str, value: pd.DataFrame) -> None:
        self.state[key] = value


class BaseAdapter:
    def __init__(self):
        pass

    def preload(self):
        pass

    def get_parent_class(self):
        for base in self.__class__.__bases__:
            return base

    def connect(self, state: GlobalState):
        raise NotImplementedError()


class Concatenator(BaseAdapter):
    def __init__(self):
        pass

    def connect(self, state: GlobalState):
        pass


class Linear(BaseModel):
    def __init__(self, blocks: List[Union[BaseModel, BaseAdapter]]):
        self.blocks = blocks
        self.config = BaseConfig(force_fit=False)
        self.state = GlobalState()

    def preload(self):
        for block in self.blocks:
            block.preload()

    def __train_model__(self, model: BaseModel, data: pd.DataFrame) -> None:
        if not model.is_fitted() or model.config.force_fit:
            model.fit(data)

    def train(self, train_dataset: pd.DataFrame) -> None:
        output = train_dataset
        for block in self.blocks:
            cls = block.get_parent_class()
            if cls == BaseAdapter:
                block.connect(self.state)
            elif cls == BaseModel:
                if block.train:
                    self.__train_model__(block, output)

                output = pd.DataFrame(
                    {
                        Const.input_col: block.predict(output[Const.input_col]),
                        Const.label_col: train_dataset[Const.label_col],
                    }
                )
                self.state.set(block.name, output)
