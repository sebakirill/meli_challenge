from typing import Callable, Dict, List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from src.data.make_dataset import ReduceMemoryUsageTransformer
from sklearn.base import BaseEstimator


def main_pipe(
    imp_cat: str,
    imp_num: str,
    cat_col: List[str],
    num_col: List[str],
    cfg_model: Dict,
    model: Callable[..., BaseEstimator],
    encoder_type: Callable,
    feature_selection: bool,
    col_selec: List[str],
) -> Pipeline:
    """
    Constructs the main pipeline for data preprocessing and modeling.

    Parameters
    ----------
    imp_cat : str
        Method for imputing missing values in categorical columns.
    imp_num : str
        Method for imputing missing values in numerical columns.
    cat_col : List[str]
        List of column names of categorical features.
    num_col : List[str]
        List of column names of numerical features.
    cfg_model : Dict
        Dictionary containing hyperparameters for the model.
    model : Callable[..., BaseEstimator]
        Callable object representing the machine learning model.
    encoder_type : Callable
        Callable object representing the encoder type.
    feature_selection : bool
        Whether to perform feature selection or not.
    col_selec : List[str]
        List of column names selected for feature selection.

    Returns
    -------
    Pipeline
        A scikit-learn pipeline object representing the main pipeline.
    """
    return Pipeline(
        [
            ("pipe_prep", ReduceMemoryUsageTransformer(feature_selection, col_selec)),
            (
                "pipe_end",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "imputing",
                                        CategoricalImputer(imputation_method=imp_cat),
                                    ),
                                    ("encoding", encoder_type()),
                                ]
                            ),
                            cat_col,
                        ),
                        (
                            "num",
                            Pipeline(
                                [
                                    (
                                        "imputing",
                                        MeanMedianImputer(imputation_method=imp_num),
                                    ),
                                ]
                            ),
                            num_col,
                        ),
                    ]
                ),
            ),
            ("model", model(**cfg_model)),
        ]
    )
