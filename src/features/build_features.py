import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features based on existing columns in the DataFrame.

    This function takes a DataFrame as input and creates several new features by combining
    or performing operations on existing columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns needed to create new features.

    Returns
    -------
    pd.DataFrame
        The DataFrame with additional features.
    """

    return (df
            .assign(fe_total_acidity=lambda df_: df_.fixed_acidity.add(df_.volatile_acidity).add(df_.citric_acid),
                    fe_free_sulfur_total_sulfur_ratio=lambda df_: df_.free_sulfur_dioxide.div(df_.total_sulfur_dioxide),
                    fe_alcohol_sugar_ratio=lambda df_: df_.alcohol.div(df_.residual_sugar),
                    fe_free_sulfur_ph_ratio=lambda df_: df_.free_sulfur_dioxide.div(df_.pH),
                    fe_alcohol_acidity_ratio=lambda df_: df_.alcohol.div(df_.fe_total_acidity),
                    fe_density_sugar_ratio=lambda df_: df_.density.div(df_.residual_sugar),
                    fe_chlorides_sugar_ratio=lambda df_: df_.chlorides.div(df_.residual_sugar),
                    fe_alcohol_times_sulfates=lambda df_: df_.alcohol.div(df_.sulphates))
            )


class CreateNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    """A transformer class to create new features in a DataFrame.

    This transformer class inherits from scikit-learn's BaseEstimator and TransformerMixin
    and can be used in a scikit-learn pipeline. It creates new features in a DataFrame by
    combining or performing operations on existing columns.

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
        self.ycol = ycol- m,

    def transform(self, X):
        """Transform the input DataFrame by creating new features.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with additional features.

        Example
        -------
        >>> transformed_data = feature_transformer.transform(original_data)
        """
        return create_new_features(X)
    
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