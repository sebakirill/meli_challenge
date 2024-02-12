from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.pipeline import Pipeline


def num_imputer(imputation_num, trial=None):
    """Create a pipeline for simple imputation of null values.

    Returns
    -------
    Pipeline
        Pipeline with the MeanMedianImputer.
    """
    num_imputer = Pipeline([
        ('imp_num', MeanMedianImputer(
            imputation_method=trial.suggest_categorical("imp_num", imputation_num)
            ))
    ])
    return num_imputer

def cat_imputer(imputation_cat, trial=None):
    """Create a pipeline for categorical imputation of null values.

    Returns
    -------
    Pipeline
        Pipeline with the CategoricalImputer.
    """
    cat_imputer = Pipeline([
        ('imp_cat', CategoricalImputer(
            imputation_method=trial.suggest_categorical("imp_cat", imputation_cat)
            ))
    ])
    return cat_imputer