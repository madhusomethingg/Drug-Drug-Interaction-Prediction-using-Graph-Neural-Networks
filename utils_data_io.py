# utils_data_io.py

import pandas as pd

def load_ddi_csv(path: str) -> pd.DataFrame:
    """
    Loads the Kaggle DDI CSV into a dataframe.
    Expected columns depend on dataset (e.g., drug_a, drug_b, label).
    """
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning: drop nulls, enforce types, etc.
    """
    df = df.dropna().reset_index(drop=True)
    return df
