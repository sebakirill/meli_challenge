from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def log_reg(C, class_weight, fit_intercept, solver, random_state, trial=None) -> LogisticRegression:
    """
    Create a Logistic Regression model.

    Parameters
    ----------
    C : dict
        Dictionary containing parameters for the regularization strength.

    class_weight : dict, list, str, or None
        Weights associated with classes in the form of a dictionary.

    fit_intercept : list
        Whether to fit an intercept term.

    solver : list
        Algorithm to use in the optimization problem.

    random_state : int
        Seed for random number generation.

    trial : optuna.trial.Trial, optional
        An optuna trial object for hyperparameter optimization.

    Returns
    -------
    LogisticRegression
        A Logistic Regression model.
    """
    return LogisticRegression(
        C=trial.suggest_float("C", **C, log=True),
        class_weight=class_weight,
        fit_intercept=trial.suggest_categorical("fit_intercept", fit_intercept),
        solver=trial.suggest_categorical("solver", solver),
        random_state=random_state,
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
) -> RandomForestClassifier:
    """
    Create a Random Forest classifier.

    Parameters
    ----------
    class_weight : dict, list, str, or None
        Weights associated with classes in the form of a dictionary.

    max_depth : dict
        Dictionary containing parameters for the maximum depth of the trees.

    min_samples_leaf : dict
        Dictionary containing parameters for the minimum number of samples required to be at a leaf node.

    min_samples_split : dict
        Dictionary containing parameters for the minimum number of samples required to split an internal node.

    n_estimators : dict
        Dictionary containing parameters for the number of trees in the forest.

    n_jobs : int
        Number of jobs to run in parallel.

    random_state : int
        Seed for random number generation.

    trial : optuna.trial.Trial, optional
        An optuna trial object for hyperparameter optimization.

    Returns
    -------
    RandomForestClassifier
        A Random Forest classifier.
    """
    return RandomForestClassifier(
        class_weight=trial.suggest_categorical("class_weight", class_weight),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", **min_samples_leaf),
        min_samples_split=trial.suggest_float("min_samples_split", **min_samples_split),
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
    trial=None,
) -> XGBClassifier:
    """
    Create an XGBoost classifier.

    Parameters
    ----------
    learning_rate : dict
        Dictionary containing parameters for the learning rate.

    min_child_weight : dict
        Dictionary containing parameters for the minimum sum of instance weight needed in a child.

    subsample : dict
        Dictionary containing parameters for the subsample ratio of the training instance.

    colsample_bytree : dict
        Dictionary containing parameters for the subsample ratio of columns when constructing each tree.

    objective : str
        Specify the learning task and the corresponding learning objective.

    scale_pos_weight : float
        Control the balance of positive and negative weights.

    n_jobs : int
        Number of parallel threads used to run xgboost.

    random_state : int
        Seed for random number generation.

    n_estimators : dict
        Dictionary containing parameters for the number of boosting rounds.

    max_depth : dict
        Dictionary containing parameters for the maximum depth of the tree.

    trial : optuna.trial.Trial, optional
        An optuna trial object for hyperparameter optimization.

    Returns
    -------
    XGBClassifier
        An XGBoost classifier.
    """

    return XGBClassifier(
        learning_rate=trial.suggest_float("learning_rate", **learning_rate),
        min_child_weight=trial.suggest_int("min_child_weight", **min_child_weight),
        subsample=trial.suggest_float("subsample", **subsample),
        colsample_bytree=trial.suggest_float("colsample_bytree", **colsample_bytree),
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        max_depth=trial.suggest_int("max_depth", **max_depth),
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
    num_leaves,
    reg_alpha,
    reg_lambda,
    min_child_samples,
    subsample_freq,
    bagging_fraction,
    bagging_freq,
    feature_fraction,
    feature_fraction_bynode,
    trial=None,
) -> LGBMClassifier:
    """
    Create a LightGBM classifier.

    Parameters
    ----------
    colsample_bytree : dict
        Parameters for the column subsampling rate of trees.

    is_unbalance : bool
        Whether to use the unbalanced label.

    learning_rate : dict
        Parameters for the learning rate.

    max_depth : dict
        Parameters for the maximum depth of the tree.

    min_child_weight : dict
        Parameters for the minimum sum of instance weight needed in a child.

    n_estimators : dict
        Parameters for the number of boosting rounds.

    n_jobs : int
        Number of parallel threads used to run LightGBM.

    objective : str
        The learning objective.

    random_state : int
        Seed for random number generation.

    subsample : dict
        Parameters for the subsample ratio of training instances.

    trial : optuna.trial.Trial, optional
        An optuna trial object for hyperparameter optimization.

    Returns
    -------
    LGBMClassifier
        A LightGBM classifier.
    """

    return LGBMClassifier(
        colsample_bytree=trial.suggest_float("colsample_bytree", **colsample_bytree),
        is_unbalance=is_unbalance,
        learning_rate=trial.suggest_float("learning_rate", **learning_rate),
        max_depth=trial.suggest_int("max_depth", **max_depth),
        min_child_weight=trial.suggest_int("min_child_weight", **min_child_weight),
        n_estimators=trial.suggest_int("n_estimators", **n_estimators),
        n_jobs=n_jobs,
        objective=objective,
        random_state=random_state,
        subsample=trial.suggest_float("subsample", **subsample),
        num_leaves=trial.suggest_int("num_leaves", **num_leaves),
        reg_alpha=trial.suggest_float("reg_alpha", **reg_alpha),
        reg_lambda=trial.suggest_float("reg_lambda", **reg_lambda),
        min_child_samples=trial.suggest_int("min_child_samples", **min_child_samples),
        subsample_freq=trial.suggest_int("subsample_freq", **subsample_freq),
        bagging_fraction=trial.suggest_float("bagging_fraction", **bagging_fraction),
        bagging_freq=trial.suggest_int("bagging_freq", **bagging_freq),
        feature_fraction=trial.suggest_float("feature_fraction", **feature_fraction),
        feature_fraction_bynode=trial.suggest_float("feature_fraction_bynode", **feature_fraction_bynode),
    )
