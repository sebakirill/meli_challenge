import hydra
from omegaconf import DictConfig
from src.data.make_dataset import get_Xs_ys

@hydra.main(config_path='../conf', config_name='config', version_base='1.3')
def train_model(cfg: DictConfig):
    print(hydra.utils.call(cfg.data.get_Xs_ys.type))
    # print(cfg.data.get_Xs_ys.type)

        


if __name__ == '__main__':
    train_model()