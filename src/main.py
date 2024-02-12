import hydra
from omegaconf import DictConfig
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import yaml
from sklearn.preprocessing import OneHotEncoder
from src.utils.utils import class_weight
from src.data.make_dataset import get_Xs_ys, ReduceMemoryUsageTransformer
from src.preprocess.encoding import FrequencyEncoder
from src.preprocess.imputing import drop_na, cat_imputer, num_imputer
from src.preprocess.feature_selection import (
    pipe_feature_selection,
    save_selected_columns,
)
from src.model.models import log_reg, random_forest, xgboost_mod, lightgmb_mod 


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train_model(cfg: DictConfig):
    X_train, X_test, y_train, y_test = hydra.utils.call(cfg.data.get_Xs_ys.type)
    hydra.utils.call(cfg.data.class_weight.type, df=y_train)
    pipe_feature_selection = hydra.utils.call(cfg.feature_selection.type)
    pipe_feature_selection.fit(X_train,y_train)
    hydra.utils.call(cfg.save_selected_columns.type, pipeline=pipe_feature_selection)
    
    def optimize_model(trial):
        
        memory_pipe = Pipeline([
            ("reduce_memory", hydra.utils.instantiate(cfg.data.reduce_memory_usage.type)),
        ])
        
        preprocess_pipe= ColumnTransformer(
            transformers=[
                ('cat', Pipeline([
                    ("imputing", hydra.utils.call(cfg.preprocess.imputing.cat_imputer.type, trial=trial)),
                    ("encoding", hydra.utils.instantiate(cfg.preprocess.encoding.type)), 
                ]), memory_pipe.fit_transform(X_train).select_dtypes('object').columns),
                ('num', Pipeline([
                    ("imputing", hydra.utils.call(cfg.preprocess.imputing.num_imputer.type, trial=trial)), 
                ]), memory_pipe.fit_transform(X_train).select_dtypes('number').columns),
            ])
        
        model_pipe = Pipeline([
            ("pipe_prep", memory_pipe),
            ("pipe_end", preprocess_pipe),
            ("model", hydra.utils.call(cfg.models.type, trial=trial))
        ])
        
        model_pipe.fit(X_train,y_train)
        y_pred = model_pipe.predict(X_test)
        return roc_auc_score(y_test, y_pred)
    
    sampler = TPESampler(seed=123)
    study = optuna.create_study(sampler = sampler, direction="maximize")
    study.optimize(optimize_model, n_trials=cfg.n_trials)

    print(f'El mejor accuracy conseguido fue: {study.best_value}')
    print(f'usando los siguientes parámetros: \n \t \t{study.best_params}')
    with open('/Users/sebastian/Proyects/meli_challenge/best_params.yaml', "w") as file:
        yaml.dump(study.best_params, file)
    
if __name__ == "__main__":
    train_model()
