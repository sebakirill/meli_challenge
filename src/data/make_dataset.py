import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


def get_Xs_ys(url :str, y_col :str) -> pd.DataFrame:
    """Load data from a CSV file, preprocess it, and split it into features and target variables.

    This function reads a CSV file from the given URL and then splits the data into features (X) 
    and target variable (y) using train_test_split.

    Parameters
    ----------
    url : str
        The URL or file path to the CSV file.

    y_col : str
        The name of the target column in the DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing X_train, X_test, y_train, and y_test.
    """

    raw = (pd
           .read_csv(url, index_col=0)
           )
    
    return train_test_split(raw.drop(columns=[y_col]), raw[y_col],
                            test_size=0.2, random_state=42, stratify=raw[y_col])


class ReduceMemoryUsageTransformer(BaseEstimator, TransformerMixin):
    """A transformer class to reduce memory usage of a DataFrame.

    This transformer class inherits from scikit-learn's BaseEstimator and TransformerMixin
    and can be used in a scikit-learn pipeline. It reduces the memory usage of a DataFrame
    by changing the data types. It is designed to be used in preprocessing steps where 
    memory optimization is required.

    Parameters
    ----------
    ycol : str, optional
        The target column in the DataFrame (default is None).

    Attributes
    ----------
    ycol : str or None
        The target column in the DataFrame.
    """

    def __init__(self, ycol=None):
        self.ycol = ycol

    def transform(self, X):
        """Transform the input DataFrame to reduce memory usage.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with reduced memory usage.
        """
        return self._reduce_memory_usage(X)
    
    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.

        y : array-like, default=None
            Ignored. This parameter exists only for compatibility.

        Returns
        -------
        self
            Returns self.
        """
        return self

    def _reduce_memory_usage(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage by changing the data types of variables.

        This function takes a pandas DataFrame as input and attempts to reduce
        its memory usage by changing the data types.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be optimized for memory usage.

        Returns
        -------
        pd.DataFrame
            The DataFrame with reduced memory usage.
        """
        
        return (X
                .assign(**{c:lambda df_, c=c:df_[c].astype('float32') for c in X.select_dtypes('float64').columns},
                        **{c:lambda df_, c=c:df_[c].astype('int32') for c in X.select_dtypes('int64').columns},
                        **{c:lambda df_, c=c:df_[c].astype('category') for c in X.select_dtypes('object').columns}
                        )
                .drop(columns= ['SIT_SITE_ID', 'PHOTO_DATE'])  
                )
    
