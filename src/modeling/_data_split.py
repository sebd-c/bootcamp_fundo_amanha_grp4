# imports
import pandas as pd
from sklearn.model_selection import train_test_split


####################################################################
def get_data_split(df: pd.DataFrame, seed: int) -> tuple:
    """
    Split dataframe into training and testing sets (80/20).
    Returns unscaled DataFrames so preprocessing can be fitted inside
    model pipelines and cross-validation folds.
    """
    X = df.drop(columns=['label'])
    y = df['label']

    X.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True, errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        stratify=y,
                                                        random_state=seed)

    return X_train, X_test, y_train, y_test
