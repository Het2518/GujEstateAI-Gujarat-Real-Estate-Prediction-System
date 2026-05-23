"""preprocess.py

Utilities to load and clean raw data.
"""
import pandas as pd

def load_raw(path):
    """Load raw CSV into DataFrame."""
    return pd.read_csv(path)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning steps and return cleaned DataFrame."""
    # placeholder: implement cleaning
    return df.copy()
