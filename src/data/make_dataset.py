import pandas as pd
import zipfile
import requests
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from typing import Tuple


def extract_zip(url: str, dst: str, member_name: str) -> pd.DataFrame:
    """Extract a member file from a zip file and read it into a pandas DataFrame.

    Parameters
    ----------
    url : str
        URL of the zip file to be downloaded and extracted.

    dst : str
        Local file path where the zip file will be written.

    member_name : str
        Name of the member file inside the zip file to be read into a DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the contents of the member file.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for non-200 status codes
    
    with open(dst, mode='wb') as fout:
        fout.write(response.content)
    
    with zipfile.ZipFile(dst) as z:
        raw = pd.read_csv(z.open(member_name), index_col=0)
        return raw

def get_Xs_ys(url: str, y_col: str, dst: str, member_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load data from a CSV file, preprocess it, and split it into features and target variables.

    This function reads a CSV file from the given URL and then splits the data into features (X) 
    and target variable (y) using train_test_split.

    Parameters
    ----------
    url : str
        The URL of the zip file containing the CSV file.

    y_col : str
        The name of the target column in the DataFrame.

    dst : str
        Local directory where the zip file will be saved and extracted.

    member_name : str
        Name of the CSV file inside the zip file to be read.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        A tuple containing X_train, X_test, y_train, and y_test.
    """

    raw = extract_zip(url=url, dst=dst, member_name=member_name)
    
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

    def __init__(self, ycol: str = None, feature_selection: bool = False, col_selec: str = None, col: str = None):
        self.feature_selection = feature_selection
        self.ycol = ycol
        self.col_selec = col_selec
        self.col = col

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        if self.feature_selection:
            return self._reduce_memory_usage(X, col=self.col_selec)
        else:
            return self._reduce_memory_usage(X, col=self.col)
    
    def fit(self, X: pd.DataFrame, y=None) -> 'ReduceMemoryUsageTransformer':
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

    def _reduce_memory_usage(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
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
                        )
                .drop(columns= col)  
                )

    
