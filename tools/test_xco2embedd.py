import hydra
from omegaconf import DictConfig, OmegaConf
import training.model_eval
import os
import wandb
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import keras


def run(cfg: DictConfig):
    wandb.login(key=cfg.wandb.key)
    config = OmegaConf.to_container(cfg, resolve=True)
    run_tags = [cfg.model.type]
    if "run_tags" in cfg:
        run_tags.extend(cfg.run_tags)
    with wandb.init(project=cfg.wandb.project_name,
                    name=cfg.exp_name, tags=run_tags, config=config) as run:
        data = model_eval.get_data_for_inversion(
            cfg.data.path.directory, cfg.data.init.path_test_ds, cfg)
        model = model_eval.get_regres_model(
            cfg.embedd_path, cfg.regres_path, data.x.eval_data.shape[1:])
        metric = model_eval.get_inv_metrics_model_on_data(model, data)
        metric["method"] = cfg.model.name
        fig = model_eval.get_summary_histo_inversion1([metric])
        wandb.log({"chart": wandb.Image(fig)})
        for key, value in metric.items():
            if isinstance(value, tf.Tensor):
                metric[key] = value.numpy().tolist()
        json.dump(metric, open("metrics.json", "w"))
        run.save(os.path.abspath("metrics.json"))


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    print("\n \n \n \n \n Run begins \n \n \n \n \n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run(cfg)
    print("\n \n \n \n \n Run ends \n \n \n \n \n")


if __name__ == "__main__":
    main()
