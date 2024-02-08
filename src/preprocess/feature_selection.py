from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.data.make_dataset import ReduceMemoryUsageTransformer
from src.preprocess.imputing import simple_imputer
from src.preprocess.encoding import freq_encoder

def pipe_feature_selection():
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
    pipe_feature_selection = Pipeline([
        ('reduce_memory', ReduceMemoryUsageTransformer()),
        ('freq_encoder', freq_encoder()),
        ('simple_imputer', simple_imputer()),
        ('xgb_class', XGBClassifier(objective="binary:logistic", enable_categorical=True))   
    ])
    return pipe_feature_selection
