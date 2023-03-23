import hydra
from omegaconf import DictConfig, OmegaConf

from src.train import train


@hydra.main(version_base=None, config_path="configs", config_name="config")
def _main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    _main()
