from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def log_reg(C, class_weight, fit_intercept, solver, random_state, trial=None):
    return LogisticRegression(
        C=trial.suggest_loguniform("C", **C),
        class_weight=trial.suggest_int("class_weight", class_weight),
        fit_intercept=trial.suggest_categorical("fit_intercept", fit_intercept),
        solver=trial.suggest_categorical("solver", solver),
        random_state=random_state
    )


def random_forest(
    class_weight,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    n_estimators,
    n_jobs,
    random_state,
    trial=None,
):
    return RandomForestClassifier(
        class_weight=trial.suggest_categorical("class_weight", class_weight),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", **min_samples_leaf),
        min_samples_split=trial.suggest_discrete_uniform(
            "min_samples_split", **min_samples_split
        ),
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        n_jobs=n_jobs,
        random_state=random_state,
    )


def xgboost_mod(
    learning_rate,
    min_child_weight,
    subsample,
    colsample_bytree,
    objective,
    scale_pos_weight,
    n_jobs,
    random_state,
    n_estimators,
    max_depth,
    min_samples_split,
    trial=None,
):

    return XGBClassifier(
        learning_rate=trial.suggest_discrete_uniform("learning_rate", **learning_rate),
        min_child_weight=trial.suggest_int("min_child_weight", **min_child_weight),
        subsample=trial.suggest_discrete_uniform("subsample", **subsample),
        colsample_bytree=trial.suggest_discrete_uniform(
            "colsample_bytree", **colsample_bytree
        ),
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_samples_split=trial.suggest_discrete_uniform(
            "min_samples_split", **min_samples_split
        ),
        objective=objective,
    )


def lightgmb_mod(
    colsample_bytree,
    is_unbalance,
    learning_rate,
    max_depth,
    min_child_weight,
    n_estimators,
    n_jobs,
    objective,
    random_state,
    subsample,
    trial=None,
):
    return LGBMClassifier(
        colsample_bytree=trial.suggest_discrete_uniform(
            "colsample_bytree", **colsample_bytree
        ),
        is_unbalance=is_unbalance,
        learning_rate=trial.suggest_discrete_uniform("learning_rate", **learning_rate),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_child_weight=trial.suggest_int("min_child_weight", **min_child_weight),
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        n_jobs=n_jobs,
        objective=objective,
        random_state=random_state,
        subsample=trial.suggest_discrete_uniform("subsample", **subsample),
    )
