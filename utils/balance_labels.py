from sklearn.utils import shuffle
import pandas as pd


def balance_labels(df: pd.DataFrame) -> pd.DataFrame:
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]

    if len(df_pos) > len(df_neg):
        df_pos = df_pos[: len(df_neg)]
    else:
        df_neg = df_neg[: len(df_pos)]

    return shuffle(pd.concat([df_pos, df_neg])).reset_index(drop=True)
