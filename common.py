"""Shared column definitions and cleaning for the heart disease project."""
import numpy as np
import pandas as pd

NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "HeartDisease"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn impossible zeros into missing values (imputed later, inside the pipeline).

    In this dataset Cholesterol == 0 and RestingBP == 0 mean "not recorded".
    """
    df = df.copy()
    for col in ("Cholesterol", "RestingBP"):
        df[col] = df[col].replace(0, np.nan)
    return df
