from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import yaml
from typing import List
from src.data.make_dataset import ReduceMemoryUsageTransformer
from src.preprocess.imputing import simple_imputer
from src.preprocess.encoding import FrequencyEncoder

def pipe_feature_selection(
    objective: str,
    enable_categorical: bool,
    imputation_num: str,
    imputation_cat: str,
    col: List[str]
) -> Pipeline:
    """
    Create a pipeline for feature selection using XGBoost.

    This pipeline includes the following steps:
    - Reduce memory usage
    - Frequency encoding
    - Simple imputation
    - XGBoost feature selection

    Parameters
    ----------
    objective : str
        The objective function for XGBoost.

    enable_categorical : bool
        Whether to enable categorical features for XGBoost.

    col : List[str]
        List of columns to be considered during feature selection.

    Returns
    -------
    Pipeline
        Pipeline for feature selection using XGBoost.
    """
    pipe_feature_selection = Pipeline(
        [
            ("reduce_memory", ReduceMemoryUsageTransformer(col=col)),
            (
                "simple_imputer", 
                simple_imputer(imputation_num=imputation_num, imputation_cat=imputation_cat)),
            ("freq_encoder", FrequencyEncoder()),
            (
                "xgb_class",
                XGBClassifier(objective=objective, enable_categorical=enable_categorical),
            ),
        ]
    )
    return pipe_feature_selection

def save_selected_columns(
    pipeline: Pipeline,
    url_file: str,
    th: int = 0
) -> None:
    """
    Extracts columns with feature importance greater than a given threshold from a trained pipeline
    and saves them to a YAML file.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline containing a feature selection step.

    url_file : str
        Path to the YAML file where the selected columns will be updated.

    th : int, default=0
        Threshold for selecting features based on their importance. Only features with
        importance greater than this threshold will be saved.

    Raises
    ------
    FileNotFoundError
        If the input YAML file specified by `input_file` does not exist.

    Notes
    -----
    This function assumes that the trained pipeline contains a step named 'xgb_class',
    which is an XGBoost classifier or regressor. The feature importances are extracted
    from this step.
    """
    # Load pipeline configuration from input YAML file
    with open(url_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Extract feature names with importance greater than the threshold
    selected_columns = [
        col
        for col, importance in zip(
            pipeline.named_steps["xgb_class"].get_booster().feature_names,
            pipeline.named_steps["xgb_class"].feature_importances_,
        )
        if importance <= th
    ]

    # Update the configuration with the selected columns
    config["type"]["col_selec"] = selected_columns

    # Save the updated configuration to the output YAML file
    with open(url_file, "w") as file:
        yaml.dump(config, file)
