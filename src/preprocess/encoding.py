import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Transformer for frequency encoding categorical variables.

    Parameters
    ----------


    Attributes
    ----------
    mapping : dict
        Mapping of category values to their frequencies for each column.
        
    cols : list
        List of column names to be frequency encoded.

    Methods
    -------
    fit(X, y=None):
        Fit the transformer by computing frequency mappings for each specified column.
    transform(X, y=None):
        Transform the DataFrame by applying frequency encoding.
    """

    def __init__(self):
        self.cols = None
        self.mapping = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer by computing frequency mappings for each specified column.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the data.
        y : Ignored
            Ignored, not used for this transformer.

        Returns
        -------
        self
            Returns the current instance of the transformer.
        """
        self.cols = X.select_dtypes('category')
        for col in self.cols:
            freq_mapping = X[col].value_counts(normalize=True).to_dict()
            self.mapping[col] = freq_mapping
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform the DataFrame by applying frequency encoding.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the data.
        y : Ignored
            Ignored, not used for this transformer.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with categorical variables encoded.
        """
        for col, freq_mapping in self.mapping.items():
            X[col + "_encoded"] = X[col].map(freq_mapping)
            X.drop(columns=[col], inplace=True)
        return X


def freq_encoder() -> Pipeline:
    """Create a pipeline with a FrequencyEncoder.

    Returns
    -------
    Pipeline
        Pipeline containing the FrequencyEncoder.
    """
    freq_encoder = Pipeline([("freq_encoder", FrequencyEncoder())])
    return freq_encoder



def one_hot_encoder() -> Pipeline:
    """Create a pipeline with a OneHotEncoder.

    Returns
    -------
    Pipeline
        Pipeline containing the OneHotEncoder.
    """
    one_hot_encoder = Pipeline([("one_hot_encoder", OneHotEncoder())])
    return one_hot_encoder

