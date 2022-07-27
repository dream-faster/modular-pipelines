from type import BaseConfig, DataType
from abc import ABC
import pandas as pd
from typing import Callable, Optional, Union, List
from runner.store import Store
from blocks.iomanager import safe_loading, safe_saving


class Element(ABC):
    id: str

    inputTypes: Union[List[DataType], DataType]
    outputType: DataType

    def children(self) -> List["Element"]:
        raise NotImplementedError()


class Block(Element):
    config: BaseConfig

    def __init__(
        self, id: Optional[str] = None, config: Optional[BaseConfig] = None
    ) -> None:
        self.id = self.__class__.__name__ if id is None else id
        self.config = (
            BaseConfig(
                force_fit=False,
                save=True,
                save_remote=False,
                preferred_load_origin=None,
            )
            if config is None
            else config
        )

        if self.inputTypes is None:
            print("inputTypes must be set")
        if self.outputType is None:
            print("outputType must be set")

    def load(self, pipeline_id: str, execution_order: int) -> int:
        self.pipeline_id = pipeline_id
        self.id += f"-{str(execution_order)}"

        model = safe_loading(pipeline_id, self.id)
        if model is not None:
            self.model = model

        return execution_order + 1

    def load_remote(self) -> None:
        pass

    def fit(
        self,
        dataset: pd.Series,
        labels: Optional[pd.Series],
    ) -> None:
        raise NotImplementedError()

    def predict(self, dataset: pd.Series) -> List:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()

    def save(self) -> None:
        safe_saving(self.model, self.pipeline_id, self.id)

    def save_remote(self) -> None:
        pass


class DataSource(Element):

    id: str

    inputTypes = DataType.Any
    outputType = DataType.Series

    def __init__(self, id: str):
        self.id = id

    def deplate(self, store: Store) -> pd.Series:
        return store.get_data(self.id)

    def load_remote(self) -> None:
        pass

    def children(self) -> List[Element]:
        return [self]
