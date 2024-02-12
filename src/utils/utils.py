import pandas as pd
import yaml


def correlation_matrix(X: pd.DataFrame, y: pd.Series, meth: str = "spearman"):
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
    cat_cols = df.select_dtypes("category").columns
    if not cat_cols.empty:
        df = pd.get_dummies(df, columns=cat_cols)

    return (
        df.corr(method=meth)
        .style.background_gradient(cmap="RdBu", vmax=1, vmin=-1)
        .set_sticky(axis="index")
    )


def check_first_buy(df: pd.DataFrame):
    """Check if any customer has made more than one purchase.

    This function groups the DataFrame by 'CUST_ID' and aggregates the 'OBJETIVO'
    column by summing the values for each customer. It then filters the resulting
    DataFrame to include only rows where the sum of 'OBJETIVO' is greater than 2,
    indicating that the customer has made more than one purchase.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing rows where a customer has made more than one purchase.
    """
    return (
        df.groupby("CUST_ID").agg({"OBJETIVO": "sum"}).loc[lambda x: x["OBJETIVO"] > 2]
    )


def class_weight(df: pd.Series, url_file: str) -> None:
    """Override class weight parameters for two models in YAML configuration files.

    This function calculates the class weights based on the provided DataFrame
    containing binary labels and updates the YAML configuration files for two XGBoost
    model.

    Parameters:
    -----------
    df : pd.Series
        Series containing binary labels for calculating class weights.

    url_files :
        File paths to the YAML configuration files of the models.

    Returns:
    --------
    None
        This function does not return any value. It updates the YAML configuration
        files in place with the calculated class weights.
    """

    val = df.value_counts()
    with open(url_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
        config["type"]["scale_pos_weight"] = round(val[0] / val[1], 2).item()
    with open(url_file, "w") as file:
        yaml.dump(config, file)
