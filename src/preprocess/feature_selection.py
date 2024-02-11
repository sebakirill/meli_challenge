from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import yaml
from src.data.make_dataset import ReduceMemoryUsageTransformer
from src.preprocess.imputing import simple_imputer
from src.preprocess.encoding import FrequencyEncoder


def pipe_feature_selection() -> Pipeline:
    """
    Create a pipeline for feature selection using XGBoost.

    This pipeline includes the following steps:
    - Reduce memory usage
    - Frequency encoding
    - Simple imputation
    - XGBoost feature selection

    Returns
    -------
    Pipeline
        Pipeline for feature selection using XGBoost.
    """
    pipe_feature_selection = Pipeline(
        [
            ("reduce_memory", ReduceMemoryUsageTransformer()),
            ("simple_imputer", simple_imputer()),
            ("freq_encoder", FrequencyEncoder()),
            (
                "xgb_class",
                XGBClassifier(objective="binary:logistic", enable_categorical=True),
            ),
        ]
    )
    return pipe_feature_selection

def save_selected_columns(pipeline: Pipeline, output_file: str, input_file: str, th: int = 0):
    """
    Extracts columns with feature importance greater than a given threshold from a trained pipeline
    and saves them to a YAML file.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline containing a feature selection step.

    output_file : str
        Path to the output YAML file where the selected columns will be saved.

    input_file : str
        Path to the input YAML file from which the pipeline configuration will be loaded.

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
    with open(input_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Extract feature names with importance greater than the threshold
    selected_columns = [
        col
        for col, importance in zip(
            pipeline.named_steps["xgb_class"].feature_names_in_,
            pipeline.named_steps["xgb_class"],
        )
        if importance > th
    ]

    # Update the configuration with the selected columns
    config["type"]["col_selec"] = selected_columns

    # Save the updated configuration to the output YAML file
    with open(output_file, "w") as file:
        yaml.dump(config, file)

