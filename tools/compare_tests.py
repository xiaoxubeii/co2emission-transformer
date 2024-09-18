import hydra
from omegaconf import DictConfig, OmegaConf
import model_eval
import os
import wandb
import json
import tensorflow as tf
import matplotlib.pyplot as plt


def _do_run(cfg: DictConfig):
    wandb.login(key=cfg.wandb.key)
    config = OmegaConf.to_container(cfg, resolve=True)
    run_tags = []
    if "run_tags" in cfg:
        run_tags.extend(cfg.run_tags)
    with wandb.init(project=cfg.wandb.project_name,
                    name=cfg.exp_name, tags=run_tags, config=config) as run:

        metrics = []
        for k, v in cfg.tests.items():
            with open(v["metric_path"], 'r') as f:
                metric = json.load(f)
                metric["method"] = v["model_name"]
                metrics.append(metric)
        fig = model_eval.get_summary_histo_inversion1(metrics)
        wandb.log({"chart": wandb.Image(fig)})
        for key, value in metric.items():
            if isinstance(value, tf.Tensor):
                metric[key] = value.numpy().tolist()
        json.dump(metric, open("metrics.json", "w"))
        run.save(os.path.abspath("metrics.json"))


@hydra.main(config_path="cfg", config_name="config")
def _run(cfg: DictConfig):
    print("\n \n \n \n \n Run begins \n \n \n \n \n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    _do_run(cfg)
    print("\n \n \n \n \n Run ends \n \n \n \n \n")


if __name__ == "__main__":
    _run()
