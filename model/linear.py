from configs.constants import Const
from .base import BaseModel
import pandas as pd
from type import BaseConfig
from typing import List, Any, Union
import numpy as np
from data.state import GlobalState
from adapter.base import BaseAdapter


class Linear(BaseModel):
    def __init__(self, blocks: List[Union[BaseModel, BaseAdapter]]):
        self.blocks = blocks
        self.config = BaseConfig(force_fit=False)
        self.state = GlobalState()
        
        self.__validity_check()

    def __validity_check(self):
        """ Check if connectors refer to previous blocks """
        block_names = ['global']
        for block in self.blocks:
            if hasattr(block,"name"):
                block_names.append(block.name)
            if hasattr(block,"keys"):
                for keys in block.keys:
                    assert keys in block_names, f"{keys} defined before models in {block_names}"
        
        """ Check if there are no duplicate names """
        assert len(block_names) == len(set(block_names)), f"Duplicate model names {block_names}"
        
    def preload(self):
        for block in self.blocks:
            block.preload()

    def __train_model__(self, model: BaseModel, data: pd.DataFrame) -> None:
        if not model.is_fitted() or model.config.force_fit:
            model.fit(data)

    def train(self, train_dataset: pd.DataFrame) -> None:
        self.state.set("global", train_dataset)

        input = train_dataset
        for block in self.blocks:
            cls = block.get_parent_class()
            if cls == BaseAdapter:
                input = block.connect(self.state)
            elif cls == BaseModel:
                if block.train:
                    self.__train_model__(block, input)

                outputs = block.predict(input[Const.input_col].astype(str))
                outputs = pd.DataFrame(
                    {
                        Const.input_col: [output[0] for output in outputs],
                        Const.prob_col: [output[1] for output in outputs],
                        Const.label_col: train_dataset[Const.label_col],
                    }
                )
                self.state.set(block.name, outputs)
                input = outputs

    
        
        