import pandas as pd

def correlation_matrix(X: pd.DataFrame, y: pd.Series, meth:  str = 'spearman'):
    """Generate a correlation matrix with dummy variables and apply styling.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with independent variables.
    y : pd.Series
        Series with the dependent variable.
    meth : str, optional 
           Method of corr.

    Returns
    -------
    pd.DataFrame
        Correlation matrix with styling.
    """

    df = pd.concat([X, y], axis=1)
    cat_cols = df.select_dtypes('category').columns
    if not cat_cols.empty:
        df = pd.get_dummies(df, columns=cat_cols)

    return (df.corr(method=meth)
            .style
            .background_gradient(cmap='RdBu', vmax=1, vmin=-1)
            .set_sticky(axis='index')
            )
