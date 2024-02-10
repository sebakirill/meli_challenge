from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def log_reg(C, fit_intercept, solver, trial=None):
    return LogisticRegression(
        C=trial.suggest_loguniform("C", **C),
        fit_intercept=trial.suggest_categorical("fit_intercept", fit_intercept),
        solver=trial.suggest_categorical("solver", solver),
        n_jobs=-1,
        random_state=123,
    )


def random_forest(n_estimators, max_depth, min_samples_split, trial=None):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_samples_split=trial.suggest_discrete_uniform(
            "min_samples_split", **min_samples_split
        ),
        n_jobs=-1,
        random_state=123,
    )


def xgboost_mod(trial=None):
    return XGBClassifier()


def lightgmb_mod(trial=None):
    return LGBMClassifier()
