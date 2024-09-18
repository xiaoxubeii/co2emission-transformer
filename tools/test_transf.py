import hydra
from omegaconf import DictConfig, OmegaConf
import training.model_eval
import os
import wandb
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from models.co2emission_transformer import emission_ensembling
from models.reg import BottomLayers


def run_test(cfg: DictConfig):
    wandb.login(key=cfg.wandb.key)
    config = OmegaConf.to_container(cfg, resolve=True)
    run_tags = [cfg.model.type]
    if "run_tags" in cfg:
        run_tags.extend(cfg.run_tags)
    with wandb.init(project=cfg.wandb.project_name,
                    name=cfg.exp_name, tags=run_tags, config=config) as run:
        data = model_eval.get_data_for_inversion(
            cfg.data.path.directory, cfg.data.init.path_test_ds, cfg)
        predictor = model_eval.get_inversion_model(
            cfg.model.predictor_path, name_w=None)
        quantifier = model_eval.get_inversion_model(
            cfg.model.quantifier_path, name_w=None)

        bottom_layers = BottomLayers(
            data.x.n_layer, data.x.eval_data.shape[-1], data.x.xco2_noisy_chans, cfg.data.init.window_length)
        x_data = data.x.eval_data[data.x.eval_data_indexes]
        x = tf.convert_to_tensor(x_data, np.float32)
        model = emission_ensembling(
            x_data.shape[1:], predictor, quantifier, bottom_layers)

        pred = tf.convert_to_tensor(model.predict(x), np.float32)
        y = tf.convert_to_tensor(data.y.eval[data.y.eval_indexes], np.float32)
        metric = model_eval.get_inv_metrics(y, pred)

        metric["method"] = cfg.model.name
        fig = model_eval.get_summary_histo_inversion1([metric])
        wandb.log({"chart": wandb.Image(fig)})
        for key, value in metric.items():
            if isinstance(value, tf.Tensor):
                metric[key] = value.numpy().tolist()
        json.dump(metric, open("metrics.json", "w"))
        run.save(os.path.abspath("metrics.json"))


@hydra.main(config_path="cfg", config_name="config")
def main_test(cfg: DictConfig):
    print("\n \n \n \n \n Run begins \n \n \n \n \n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run_test(cfg)
    print("\n \n \n \n \n Run ends \n \n \n \n \n")


if __name__ == "__main__":
    main_test()
