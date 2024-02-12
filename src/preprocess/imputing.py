import pandas as pd
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class DropNaTransformer(BaseEstimator, TransformerMixin):
    """Transformer that removes columns with a specified null value threshold.

    Parameters
    ----------
    th : float, optional (default=0.2)
        Threshold for the proportion of null values in a column.

    Attributes
    ----------
    th : float
        Current threshold configured for the proportion of null values.

    Methods
    -------
    transform(X):
        Transforms the DataFrame by removing columns with null values based on the configuration.
    fit(X, y=None):
        Fits the transformer by storing the list of columns to be dropped.
    _drop_col(X):
        Generates a list of columns to be dropped based on the null value threshold.

    """

    def __init__(self, th: float = 0.2):
        """Initializes the transformer.

        Parameters
        ----------
        th : float, optional (default=0.2)
            Threshold for the proportion of null values in a column.
        """
        self.th = th
        self.columns_to_drop = None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the DataFrame by removing columns with null values based on the configuration.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Resulting DataFrame after removing columns with null values.
        """
        return X.drop(columns=self.columns_to_drop)

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the transformer by storing the list of columns to be dropped.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        y : Ignored
            Ignored, not used for this transformer.

        Returns
        -------
        self
            Returns the current instance of the transformer.
        """
        self.columns_to_drop = X.columns[X
                                         .isnull()
                                         .sum()
                                         .div(X.shape[1])
                                         .gt(self.th)
                                         ]
        return self

def drop_na(trial=None):
    """Create a pipeline for dropping columns with null values.

    Returns
    -------
    Pipeline
        Pipeline with the DropNaTransformer.
    """
    drop_na = Pipeline([
        ('drop_na', DropNaTransformer())
    ])
    return drop_na

def num_imputer(imputation_num, trial=None):
    """Create a pipeline for simple imputation of null values.

    Returns
    -------
    Pipeline
        Pipeline with the MeanMedianImputer.
    """
    num_imputer = Pipeline([
        ('imp_num', MeanMedianImputer(
            imputation_method=trial.suggest_categorical("imp_num", imputation_num)
            ))
    ])
    return num_imputer

def cat_imputer(imputation_cat, trial=None):
    """Create a pipeline for categorical imputation of null values.

    Returns
    -------
    Pipeline
        Pipeline with the CategoricalImputer.
    """
    cat_imputer = Pipeline([
        ('imp_cat', CategoricalImputer(
            imputation_method=trial.suggest_categorical("imp_cat", imputation_cat)
            ))
    ])
    return cat_imputer