
import hydra
from omegaconf import DictConfig, OmegaConf
import model_eval
import os
import wandb
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Data import Data_eval
import tools.matplotlib_functions as mympf
import keras
from tools.plot_image import plot_image_from_mae
from training.model_training import Model_training_manager


def run(cfg: DictConfig):
    model_trainer = Model_training_manager(cfg)


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    print("\n \n \n \n \n Run begins \n \n \n \n \n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run(cfg)
    print("\n \n \n \n \n Run ends \n \n \n \n \n")


if __name__ == "__main__":
    main()
