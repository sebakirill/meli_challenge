from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def log_reg(C, fit_intercept, solver, random_state, class_weight, trial=None):
    return LogisticRegression(
        C=trial.suggest_loguniform("C", **C),
        fit_intercept=trial.suggest_categorical("fit_intercept", fit_intercept),
        solver=trial.suggest_categorical("solver", solver),
        random_state=trial.suggest_int("random_state", **random_state),
        class_weight=trial.suggest_int("class_weight", class_weight)
    )


def random_forest(
    n_estimators,
    max_depth,
    min_samples_split,
    bootstrap,
    random_state,
    n_jobs,
    class_weight,
    trial=None,
):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_samples_split=trial.suggest_discrete_uniform(
            "min_samples_split", **min_samples_split
        ),
        n_jobs=trial.suggest_int("n_jobs", **n_jobs),
        random_state=trial.suggest_int("random_state", **random_state),
        bootstrap=trial.suggest_categorical("bootstrap", **bootstrap),
        class_weight=trial.suggest_int("class_weight", class_weight)
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
    learning_rate,
    min_child_weight,
    subsample,
    colsample_bytree,
    objective,
    is_unbalance,
    n_jobs,
    random_state,
    n_estimators,
    max_depth,
    min_samples_split,
    trial=None,
):
    return LGBMClassifier(
        learning_rate=trial.suggest_discrete_uniform("learning_rate", **learning_rate),
        min_child_weight=trial.suggest_int("min_child_weight", **min_child_weight),
        subsample=trial.suggest_discrete_uniform("subsample", **subsample),
        colsample_bytree=trial.suggest_discrete_uniform(
            "colsample_bytree", **colsample_bytree
        ),
        objective=trial.suggest_categorical("objective", objective),
        is_unbalance=trial.suggest_categorical("is_unbalance", is_unbalance),
        n_jobs=trial.suggest_int("n_jobs", n_jobs),
        random_state=trial.suggest_int("random_state", random_state),
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_samples_split=trial.suggest_discrete_uniform(
            "min_samples_split", **min_samples_split
        ),
    )
