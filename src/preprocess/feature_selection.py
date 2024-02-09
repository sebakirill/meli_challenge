from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import yaml
from src.data.make_dataset import ReduceMemoryUsageTransformer
from src.preprocess.imputing import simple_imputer
from src.preprocess.encoding import freq_encoder


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
            ("freq_encoder", freq_encoder()),
            (
                "xgb_class",
                XGBClassifier(objective="binary:logistic", enable_categorical=True),
            ),
        ]
    )
    return pipe_feature_selection


def save_selected_columns(pipeline, output_file, th: int = 0):
    """
    Extracts columns with feature importance greater than 0 from a trained pipeline
    and saves them to a YAML file.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline containing a feature selection step.

    output_file : str
        Path to the output YAML file where the selected columns will be saved.
    """

    selected_columns = [
        col
        for col, importance in zip(
            pipeline.named_steps["xgb"].feature_names_in_,
            pipeline.named_steps["xgb_class"],
        )
        if importance > th
    ]

    with open(output_file, "w") as file:
        yaml.dump(selected_columns, file)
