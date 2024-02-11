import hydra
from omegaconf import DictConfig
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.utils.utils import class_weight
from src.data.make_dataset import get_Xs_ys, ReduceMemoryUsageTransformer
from src.preprocess.imputing import simple_imputer, drop_na
from src.preprocess.encoding import FrequencyEncoder
from src.preprocess.feature_selection import (
    pipe_feature_selection,
    save_selected_columns,
)



@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train_model(cfg: DictConfig):
    X_train, X_test, y_train, y_test = hydra.utils.call(cfg.data.get_Xs_ys.type)
    hydra.utils.call(cfg.data.class_weight.type, df=y_train)
    # pipe_feature_selection = hydra.utils.call(cfg.feature_selection.type)
    # pipe_feature_selection.fit(X_train,y_train)
    # hydra.utils.call(cfg.save_selected_columns.type, pipeline=pipe_feature_selection)


    # def optimize_model(trial):
    #     preprocess_pipe = Pipeline([
    #         ("reduce_memory", hydra.utils.instantiate(cfg.data.reduce_memory_usage.type)),
    #         ("imputer", hydra.utils.call(cfg.prerocess.encoding.type, trial=trial)),
    #         ("encoding", hydra.utils.instantiate(cfg.preprocess.imputing.type, trial=trial))
    #     ])
    #     model_pipe = Pipeline([
    #         ("pipe_prep", preprocess_pipe),
    #         ("model", hydra.utils.call(cfg.models.type, trial=trial))
    #     ])
    #     model_pipe.fit(X_train,y_train)
    #     y_pred = model_pipe.predict(X_test)
    #     return accuracy_score(y_test, y_pred)





    # pipe_model.fit(X_train, y_train)
    
if __name__ == "__main__":
    train_model()
