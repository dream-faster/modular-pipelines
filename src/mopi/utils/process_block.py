from typing import List, Optional, Union

import pandas as pd


def process_block(
    block: Union["DataSource", "Pipeline", "Concat"],
    store: "Store",
    plugins: List["Plugin"],
    train: bool,
) -> pd.Series:
    from mopi.blocks.base import DataSource
    from mopi.blocks.concat import Concat
    from mopi.blocks.pipeline import Pipeline

    if isinstance(block, DataSource) or isinstance(block, Concat):
        return block.deplate(store, plugins, train=train)
    elif isinstance(block, Pipeline):
        if not block.is_fitted() and train == True:
            block.fit(store, plugins)
        return block.predict(store, plugins)
    else:
        raise ValueError("Expected DataBlock or Pipeline")
