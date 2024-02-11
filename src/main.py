import hydra
from omegaconf import DictConfig
import pandas as pd
from src.data.make_dataset import get_Xs_ys, ReduceMemoryUsageTransformer
from src.preprocess.imputing import simple_imputer, drop_na
from src.preprocess.encoding import one_hot_encoder, freq_encoder
from src.preprocess.feature_selection import (
    pipe_feature_selection,
    save_selected_columns,
)
from sklearn.pipeline import Pipeline


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train_model(cfg: DictConfig):
    # hydra.utils.call(cfg.data.extract_zip.type)
    X_train, X_test, y_train, y_test = hydra.utils.call(cfg.data.get_Xs_ys.type)
    pipe_feature_selection = hydra.utils.call(cfg.feature_selection.type)
    pipe_feature_selection.fit(X_train,y_train)

if __name__ == "__main__":
    train_model()
