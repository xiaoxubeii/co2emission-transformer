# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys
import matplotlib.pyplot as plt
import keras_nlp
import os
import itertools
import tools.matplotlib_functions as mympf
from Data import Data_eval
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import xarray as xr
from keras import ops
from cmcrameri import cm
from omegaconf import OmegaConf
from scipy.optimize import differential_evolution
from include.loss import pixel_weighted_cross_entropy
from model_training import Model_training_manager


# Segmentation


def get_data_for_segmentation(
    dir_res: str,
    path_eval_nc: str,
    ds_inds: dict = dict(),
    region_extrapol: bool = True,
) -> Data_eval:
    """Prepare Data object with name_dataset, and train or test mode."""
    from cfg.convert_cfg_to_yaml import save_myyaml_from_mycfg

    if not os.path.exists(os.path.join(dir_res, "config.yaml")):
        save_myyaml_from_mycfg(dir_res)
    cfg = OmegaConf.load(os.path.join(dir_res, "config.yaml"))

    data = Data_eval(path_eval_nc)
    data.prepare_input(
        cfg.data.input.chan_0,
        cfg.data.input.chan_1,
        cfg.data.input.chan_2,
        cfg.data.input.chan_3,
        cfg.data.input.chan_4,
    )
    data.prepare_output_segmentation(
        curve=cfg.data.output.curve,
        min_w=cfg.data.output.min_w,
        max_w=cfg.data.output.max_w,
        param_curve=cfg.data.output.param_curve,
    )
    return data


def get_segmentation_model(
    dir_res: str,
    name_w: str = "w_best.h5",
    optimiser: str = "adam",
    loss=pixel_weighted_cross_entropy,
):
    """Get segmentation neural network model and compile it with pixel_weighted_cross_entropy loss."""
    model = tf.keras.models.load_model(
        os.path.join(dir_res, name_w), compile=False)
    model.compile(optimiser, loss=loss)
    return model


def get_wbce(y_test: tf.Tensor, pred_test: tf.Tensor) -> np.ndarray:
    """Get wbce given y_test and pred_test."""
    all_wbce = pixel_weighted_cross_entropy(y_test, pred_test, reduction=False)
    all_wbce = np.mean(all_wbce, axis=(1, 2))
    return all_wbce


def get_wbce_model_on_data(model: tf.keras.Model, data: Data_eval) -> np.ndarray:
    """Get wbce scores by segmentation model applied on data."""
    x = tf.convert_to_tensor(data.x.eval, np.float32)
    pred = tf.convert_to_tensor(model.predict(x), np.float32)
    y = tf.convert_to_tensor(data.y.eval, np.float32)
    all_wbce = get_wbce(y, pred)
    return all_wbce


def get_nwbce_model_on_data(model: tf.keras.Model, data: Data_eval) -> np.ndarray:
    """Get nwbce scores by segmentation model applied on data."""
    all_cnn_wbce = get_wbce_model_on_data(model, data)
    b1_all_wbce = get_b1_seg_wbce(
        tf.convert_to_tensor(data.y.eval, np.float32))
    all_cnn_nwbce = all_cnn_wbce / b1_all_wbce
    return all_cnn_nwbce


# neutral baseline functions


def get_mean_loss(params, y_test: tf.Tensor, pred_test: tf.Tensor) -> float:
    """Get mean wbce between y_test and pred_test given shift_to_proba with params."""
    proba_min, proba_max = params
    current_pred_test = shift_to_proba(pred_test, proba_max, proba_min)
    wbce = get_wbce(y_test, current_pred_test)
    return np.mean(wbce)


def shift_to_proba(y_pred, proba_max: np.float32, proba_min: np.float32):
    """Shift from a boolean to a probability map: 1 to proba_max, 0 to proba_min."""
    y_pred = np.where(y_pred == 1, proba_max, proba_min)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    return y_pred


def get_b1_seg_pred(y: tf.Tensor):
    """Get neutral reference/baseline (b1) segmentation predictions."""
    b1_pred = 0.1 * tf.ones(shape=y.shape)
    res = differential_evolution(
        get_mean_loss, args=(y, b1_pred), bounds=[[0, 1], [0, 1]]
    )
    [proba_min, proba_max] = res["x"]
    shifted_b1_pred = shift_to_proba(b1_pred, proba_max, proba_min)
    return shifted_b1_pred


def get_b1_seg_wbce(y: tf.Tensor) -> np.ndarray:
    """Get wbce for y and neutral reference/baseline (b1) segmentation predictions."""
    b1_pred = get_b1_seg_pred(y)
    wbce = get_wbce(y, b1_pred)
    return wbce


# plot functions


def plot_segmentation_examples(
    data: Data_eval,
    cnn_nwbce: np.ndarray,
    model: tf.keras.Model,
    list_idx: list = [],
    list_ds_idx: list = [],
    proba_plume: float = 0,
    no2=False,
):
    """Plot examples of {input / truth / output} of the CNN model."""

    if not list_idx:
        [idx0, ds_idx0] = draw_idx(cnn_nwbce, data.ds)
        [idx1, ds_idx1] = draw_idx(cnn_nwbce, data.ds)
        [idx2, ds_idx2] = draw_idx(cnn_nwbce, data.ds)
        list_idx = [idx0, idx1, idx2]
        list_ds_idx = [ds_idx0, ds_idx1, ds_idx2]

    N_idx = len(list_idx)

    N_cols = 3
    if proba_plume > 0:
        N_cols = N_cols + 1
    if no2:
        N_cols = N_cols + 1

    mympf.setMatplotlibParam()
    plt.viridis()
    fig, axs = mympf.set_figure_axs(
        N_idx,
        N_cols,
        wratio=0.35,
        hratio=0.75,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.001,
        pad_h_ext=0.2,
        pad_h_int=0.15,
    )

    ims = [None] * (N_idx * N_cols)
    caxs = [None] * (N_idx * N_cols)
    cbars = [None] * (N_idx * N_cols)

    Ny = int(data.x.eval.shape[1])
    Nx = int(data.x.eval.shape[2])
    for ax in axs:
        ax.set_xticks([0, int(Ny / 4), int(Ny / 2), int(Ny * 3 / 4), Ny])
        ax.set_yticks([0, int(Nx / 4), int(Nx / 2), int(Nx * 3 / 4), Nx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, idx in enumerate(list_idx):

        x_idx = data.x.eval[idx]

        cur_row = 0

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(data.x.eval[idx, :, :, 0]), origin="lower"
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(
            ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(data.y.eval[idx]), origin="lower")
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(
            ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(model(tf.expand_dims(data.x.eval[idx], 0))[0]),
            vmin=0,
            vmax=1,
            cmap=cm.cork,
            origin="lower",
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(
            ims[i_ax], caxs[i_ax], orientation="vertical")

        if no2:
            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.squeeze(data.x.eval[idx, :, :, -1]), origin="lower"
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

        if proba_plume > 0:
            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.where(
                    np.squeeze(model(tf.expand_dims(data.x.eval[idx], 0))[0])
                    > proba_plume,
                    1,
                    0,
                ),
                origin="lower",
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

    list_pd_t_idx = []
    list_cnn_nwbce = []
    for idx, ds_idx in enumerate(list_ds_idx):
        list_pd_t_idx.append(pd.Timestamp(ds_idx.time.values))
        list_cnn_nwbce.append(cnn_nwbce[list_idx[idx]])

    for i, (pd_t_idx, loss_idx) in enumerate(
        zip(
            list_pd_t_idx,
            list_cnn_nwbce,
        )
    ):
        axs[i * N_cols].set_ylabel(
            f"[{pd_t_idx.month:02d}-{pd_t_idx.day:02d} {pd_t_idx.hour:02d}:00], n_wbce={loss_idx: .3f}"
        )

    axs[0].set_title("XCO2 field")
    axs[1].set_title("Targetted plume")
    axs[2].set_title("CNN segmentation")

    cbars[0].ax.set_title("[ppmv]")
    cbars[1].ax.set_title("[weight. bool.]")
    cbars[2].ax.set_title("[proba.]")


# Inversion
#  Get functions


def get_data_for_inversion(data_dir: str, path_eval_nc: str, cfg: OmegaConf = None
                           ) -> Data_eval:
    """Prepare Data_eval object with name_dataset."""

    if cfg is None:
        cfg = OmegaConf.load(os.path.join(dir_res, "config.yaml"))
    if 'window_length' not in cfg.data.init:
        data = Data_eval(data_dir, path_eval_nc, 0, 0)
    else:
        data = Data_eval(data_dir, path_eval_nc, cfg.data.init.window_length,
                         cfg.data.init.shift)

    data.prepare_input(
        cfg.data.input.chan_0,
        cfg.data.input.chan_1,
        cfg.data.input.chan_2,
        cfg.data.input.chan_3,
        cfg.data.input.chan_4,
    )
    data.prepare_output_inversion(cfg.data.output.N_emissions)
    return data


def get_regres_model(embedd_path, regres_path, input_shape):
    regres_model = keras.models.load_model(regres_path)
    mae_model = keras.models.load_model(embedd_path)
    mae_model.patch_encoder.downstream = True

    quanti_model = regres_model.quantifier
    embedd_model = keras.Sequential(
        [mae_model.patch_layer, mae_model.patch_encoder, mae_model.encoder])
    inputs = keras.Input(input_shape)
    x = embedd_model(inputs)
    x = keras.layers.Flatten()(x)
    outputs = quanti_model(x)
    return keras.Model(inputs, outputs)


def get_inversion_model(
    dir_res: str,
    name_w: str = "w_best.keras",
    optimiser: str = "adam",
    loss=tf.keras.losses.MeanAbsoluteError(),
):
    if name_w:
        model_path = os.path.join(dir_res, name_w)
    else:
        model_path = dir_res
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimiser, loss=loss)
    return model


def get_inversion_model_from_weights(dir_res, name_w="w_last.weights.h5", cfg: OmegaConf = None):
    if cfg is None:
        cfg = OmegaConf.load(os.path.join(dir_res, "config.yaml"))
    model = Model_training_manager(cfg).model
    if name_w:
        model_path = os.path.join(dir_res, name_w)
    else:
        model_path = dir_res
    model.load_weights(model_path)
    return model


def print_inv_metrics(metrics, message=""):
    """Print MAE and MAPE inv metrics."""
    print(message)
    print("MAE", np.mean(metrics["mae"]), np.median(metrics["mae"]))
    print("MAPE", np.mean(metrics["mape"]), np.median(metrics["mape"]))


def get_inv_metrics_pred_from_ensemble_paths(list_paths_model, path_ds_nc, name_w="w_best.h5"):
    """Get inv metrics from ensemble of paths to models."""
    data = get_data_for_inversion(
        list_paths_model[0],
        path_ds_nc,
    )
    y = tf.convert_to_tensor(data.y.eval, np.float32)
    pred = tf.zeros_like(y)
    for path_model in list_paths_model:
        model = get_inversion_model(path_model, name_w=name_w)
        x = tf.convert_to_tensor(data.x.eval, np.float32)
        cur_pred = tf.convert_to_tensor(model.predict(x), np.float32)
        pred += cur_pred
        print_inv_metrics(get_inv_metrics(y, cur_pred), message=path_model)
    pred = pred / len(list_paths_model)
    metrics = get_inv_metrics(y, pred)
    print_inv_metrics(metrics, message="\nEnsemble")
    return {"metrics": metrics, "pred": pred, "data": data}


def get_inv_metrics(y: tf.Tensor, pred: tf.Tensor):
    """Get inversion metrics between predictions and truth."""
    f_mae = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.losses.Reduction.NONE)
    all_mae = f_mae(y, pred)
    f_mape = tf.keras.losses.MeanAbsolutePercentageError(
        reduction=tf.losses.Reduction.NONE
    )
    all_mape = f_mape(y, pred)
    return {"mae": all_mae, "mape": all_mape}


def get_inv_metrics_model_on_data(model: tf.keras.Model, data: Data_eval, sample_num=0) -> dict:
    """Get inversion scores by inversion model applied on data."""
    if sample_num > 0:
        data_index = data_index[:sample_num]
    x = tf.convert_to_tensor(
        data.x.eval_data[data.x.eval_data_indexes], np.float32)
    pred = tf.convert_to_tensor(model.predict(x), np.float32)
    y = tf.convert_to_tensor(data.y.eval[data.y.eval_indexes], np.float32)
    if sample_num > 0:
        y = y[:sample_num]
    # if data.window_length > 0:
    #     y = tf.reshape(y, (y.shape[0]*y.shape[1], y.shape[2]))
    #     pred = tf.reshape(pred, (pred.shape[0]*pred.shape[1], pred.shape[2]))
    return get_inv_metrics(y, pred)


def get_inv_mean_loss(data: Data_eval) -> dict:
    """Get mean inventory for inversion between y and mean predictions."""
    y = tf.convert_to_tensor(data.y.eval, np.float32)
    pred = tf.math.reduce_mean(y) * tf.ones_like(y, np.float32)
    return get_inv_metrics(y, pred)


def get_inv_metrics_from_paths(path_model: str, path_ds_nc: str):
    """Get inversion scores using only paths as inputs."""
    data = get_data_for_inversion(
        path_model,
        path_ds_nc,
    )
    model = get_inversion_model(path_model, name_w="w_best.h5")
    metrics = get_inv_metrics_model_on_data(model, data)
    print_inv_metrics(metrics, message=path_model)
    return metrics


# plot functions
def draw_idx(
    all_scores: np.ndarray, ds: xr.Dataset, interval: list = [], idx: int = -1
) -> list:
    """Draw a specific field/plume/emiss index to plot given potential interval."""
    if idx > 0:
        pass
    elif interval:
        quantiles = np.quantile(all_scores, interval)
        idx = np.random.choice(
            np.argwhere(
                (quantiles[0] < all_scores) & (all_scores < quantiles[1])
            ).flatten()
        )
    else:
        idx = int(np.random.uniform(0, all_scores.shape[0]))

    ds_idx = ds.isel(idx_img=idx)
    return [idx, ds_idx]


def get_summary_histo_inversion1(metrics):
    df_mae = [pd.DataFrame({"loss": m["mae"], "method": f'{m["method"]}: \nmean {np.mean(m["mae"]):.2f}, median {np.median(m["mae"]):.2f}, total {len(m["mae"]):d}'})
              for m in metrics]
    df_mape = [pd.DataFrame(
        {"loss": m["mape"], "method": f'{m["method"]}: \nmean {np.mean(m["mape"]):.2f}, median {np.median(m["mape"]):.2f} total {len(m["mae"]):d}'}) for m in metrics]

    df_mae = pd.concat(df_mae)
    df_mape = pd.concat(df_mape)
    N_rows = 1
    N_cols = 2
    mympf.setMatplotlibParam()
    plt.viridis()
    fig, axs = mympf.set_figure_axs(
        N_rows,
        N_cols,
        wratio=0.35,
        hratio=1,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.3,
        pad_h_ext=0.7,
        pad_h_int=0.35,
    )

    sns.kdeplot(
        data=df_mae,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[0],
    )
    sns.kdeplot(
        data=df_mape,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[1],
    )
    titles = [
        "Mean absolute error",
        "Mean absolute percentage error",
    ]

    for i_ax, ax in enumerate(axs):
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_xlabel(titles[i_ax])
        plt.setp(ax.get_legend().get_texts(), fontsize='4')  # for legend text
        ax.get_legend().set_loc("upper center")
        ax.get_legend().set_bbox_to_anchor((0.75, 1.35))
        ax.get_legend().set_title(None)

    return fig


def get_summary_histo_inversion(
    model: tf.keras.Model, data: Data_eval, dir_save: str = "None"


) -> None:
    """Get various histograms summing up the inversion results."""
    metrics = get_inv_metrics_model_on_data(model, data)
    # mean_metrics = get_inv_mean_loss(data)

    df_mae = pd.DataFrame({"loss": metrics["mae"], "method": "CNN"})
    # df_mae_2 = pd.DataFrame({"loss": mean_metrics["mae"], "method": "mean"})
    # df_mae = pd.concat([df_mae_1, df_mae_2])

    df_mape = pd.DataFrame({"loss": metrics["mape"], "method": "CNN"})
    # df_mape_2 = pd.DataFrame({"loss": mean_metrics["mape"], "method": "mean"})
    # df_mape = pd.concat([df_mape_1, df_mape_2])

    # pred = np.squeeze(model.predict(
    #     tf.convert_to_tensor(data.x.eval, np.float32)))
    # y = data.y.eval[:, -1]
    # df_emiss_1 = pd.DataFrame({"emiss": y, "origin": "truth"})
    # df_emiss_2 = pd.DataFrame({"emiss": pred, "origin": "prediction"})
    # df_emiss = pd.concat([df_emiss_1, df_emiss_2])

    N_rows = 1
    N_cols = 2
    mympf.setMatplotlibParam()
    plt.viridis()
    fig, axs = mympf.set_figure_axs(
        N_rows,
        N_cols,
        wratio=0.35,
        hratio=0.75,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.3,
        pad_h_ext=0.3,
        pad_h_int=0.35,
    )

    sns.kdeplot(
        data=df_mae,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[0],
    )
    sns.kdeplot(
        data=df_mape,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[1],
    )
    # sns.kdeplot(
    #     data=df_emiss,
    #     x="emiss",
    #     common_norm=True,
    #     hue="origin",
    #     color="firebrick",
    #     fill=True,
    #     alpha=0.2,
    #     ax=axs[2],
    # )
    # sns.kdeplot(pred / y, color="firebrick", fill=True, alpha=0.2, ax=axs[3])

    titles = [
        "Mean absolute error",
        "Mean absolute percentage error",
        # "Emission rate",
        # "Prediction/Truth",
    ]

    for i_ax, ax in enumerate(axs):
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_xlabel(titles[i_ax])

    if dir_save != "None":
        plt.savefig(os.path.join(dir_save, "summary_inv.png"))

    return fig


def plot_inversion_examples(
    data: Data_eval,
    all_mae: np.ndarray,
    all_mape: np.ndarray,
    model: tf.keras.Model,
    list_idx: list = [],
    list_ds_idx: list = [],
    proba_plume: float = 0,
    fourth_col=False,
    window_length=0,
):
    """Plot examples of {input / truth / output} of the CNN model."""
    if not list_idx:
        if window_length > 0:
            [idx, ds_idx] = draw_idx(all_mae, data.ds)
            list_idx = [idx]*window_length
            list_ds_idx = [ds_idx]*window_length
        else:
            [idx0, ds_idx0] = draw_idx(all_mae, data.ds)
            [idx1, ds_idx1] = draw_idx(all_mae, data.ds)
            [idx2, ds_idx2] = draw_idx(all_mae, data.ds)
            list_idx = [idx0, idx1, idx2]
            list_ds_idx = [ds_idx0, ds_idx1, ds_idx2]

    N_idx = len(list_idx)
    N_cols = 3
    if proba_plume > 0:
        N_cols = N_cols + 1
    if fourth_col:
        N_cols = N_cols + 1

    mympf.setMatplotlibParam()
    plt.viridis()
    fig, axs = mympf.set_figure_axs(
        N_idx,
        N_cols,
        wratio=0.35,
        hratio=0.75,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.001,
        pad_h_ext=0.2,
        pad_h_int=0.25,
    )

    ims = [None] * (N_idx * N_cols)
    caxs = [None] * (N_idx * N_cols)
    cbars = [None] * (N_idx * N_cols)

    Ny = int(data.x.eval.shape[1])
    Nx = int(data.x.eval.shape[2])
    for ax in axs:
        ax.set_xticks([0, int(Ny / 4), int(Ny / 2), int(Ny * 3 / 4), Ny])
        ax.set_yticks([0, int(Nx / 4), int(Nx / 2), int(Nx * 3 / 4), Nx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def _plot_data(data, list_idx):
        for i, idx in enumerate(list_idx):
            cur_row = 0
            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.squeeze(data[idx, :, :, 0]), origin="lower"
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.squeeze(data[idx, :, :, 1]), origin="lower"
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.squeeze(data[idx, :, :, 2]), origin="lower"
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(
                ims[i_ax], caxs[i_ax], orientation="vertical")

            if fourth_col:
                i_ax = cur_row + i * N_cols
                cur_row += 1
                ims[i_ax] = axs[i_ax].imshow(
                    np.squeeze(data[idx, :, :, 3]), origin="lower"
                )
                caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
                cbars[i_ax] = plt.colorbar(
                    ims[i_ax], caxs[i_ax], orientation="vertical")

    if window_length > 0:
        _plot_data(data.x.eval[list_idx[0]], range(window_length))
    else:
        _plot_data(data.x.eval, list_idx)

    list_pd_t_idx = []
    list_all_scores = []
    for idx, ds_idx in enumerate(list_ds_idx):
        list_pd_t_idx.append(pd.Timestamp(ds_idx.time.values))
        list_all_scores.append(all_mae[list_idx[idx]])

    for i, (pd_t_idx, loss_idx) in enumerate(
        zip(
            list_pd_t_idx,
            list_all_scores,
        )
    ):
        axs[i * N_cols].set_ylabel(
            f"[{pd_t_idx.month:02d}-{pd_t_idx.day:02d} {pd_t_idx.hour:02d}:00]"
        )

    for i, idx in enumerate(list_idx):
        axs[1 + i * N_cols].set_title(
            f"time: {pd.Timestamp(list_ds_idx[i].time.values)},   "
            f"mae: {all_mae[idx].numpy():.3f},   "
            f"mape: {all_mape[idx].numpy():.3f},   "
            f"truth: {data.y.eval[idx][0]:.3f},   "
            f"pred: {model.predict(data.x.eval[idx:idx+1])[0][0]:.3f},   "
        )


def channel_permutation_importance(dir_model: str, path_eval_nc: str, size_max_combi=1):
    """Get importances of each channel for a model and a dataset at path."""
    data = get_data_for_inversion(
        dir_model,
        path_eval_nc,
    )

    model = get_inversion_model(dir_model, name_w="w_best.h5")
    X = data.x.eval
    y = data.y.eval
    loss_function = tf.keras.losses.MeanAbsolutePercentageError()

    baseline_predictions = model(X)
    baseline_loss = loss_function(y, baseline_predictions).numpy()
    print("baseline", baseline_loss)

    my_list = [i for i in range(X.shape[-1])]
    combinations = []
    for r in range(1, size_max_combi + 1):
        for combo in itertools.combinations(my_list, r):
            combinations.append(combo)

    importances = np.zeros(len(combinations))

    for i, perm in enumerate(combinations):
        X_permuted = np.copy(X)
        for p in perm:
            X_permuted[:, :, :, p] = np.random.permutation(X[:, :, :, p])

        # Compute predictions with permuted channel
        permuted_predictions = model(X_permuted)
        permuted_loss = loss_function(y, permuted_predictions).numpy()
        print("perm", perm, "permuted_loss", permuted_loss)

        # Compute feature importance for channel
        importances[i] = baseline_loss - permuted_loss

    return importances


def build_df_perf_inv(metrics):
    """Build dataframe of inversion performances."""

    metrics_none = metrics["none"]
    metrics_seg_pred_no2 = metrics["seg_pred_no2"]
    metrics_no2 = metrics["no2"]
    metrics_cs = metrics["cs"]

    mape_col = "Relative error (%)"
    second_col = "Add. input:"
    df_mape_none = pd.DataFrame(
        {mape_col: metrics_none["mape"], second_col: "No additional input"}
    )
    df_mape_seg = pd.DataFrame(
        {mape_col: metrics_seg_pred_no2["mape"], second_col: "Segmentation"}
    )
    df_mape_no2 = pd.DataFrame(
        {mape_col: metrics_no2["mape"], second_col: "XNO2"})
    df_mape_cs = pd.DataFrame(
        {mape_col: metrics_cs["mape"], second_col: "CSF"}
    )
    df_mape_cs[mape_col] = df_mape_cs[mape_col].apply(
        lambda x: 200 if x > 200 else x)
    df_mape = pd.concat([df_mape_none, df_mape_seg, df_mape_no2, df_mape_cs])

    mae_col = "Absolute error (Mt/yr)"
    second_col = "Add. input:"
    df_mae_none = pd.DataFrame(
        {mae_col: metrics_none["mae"], second_col: "No additional input"}
    )
    df_mae_seg = pd.DataFrame(
        {mae_col: metrics_seg_pred_no2["mae"], second_col: "Segmentation"}
    )
    df_mae_no2 = pd.DataFrame(
        {mae_col: metrics_no2["mae"], second_col: "XNO2"})
    df_mae_cs = pd.DataFrame(
        {mae_col: metrics_cs["mae"], second_col: "CSF"}
    )
    df_mae_cs[mae_col] = df_mae_cs[mae_col].apply(
        lambda x: 30 if x > 30 else x)
    df_mae = pd.concat([df_mae_none, df_mae_seg, df_mae_no2, df_mae_cs])

    df_mape_groupby = df_mape.groupby("Add. input:")
    desc = df_mape_groupby.describe()
    desc = desc.drop((mape_col, "count"), axis=1)
    desc = desc.drop((mape_col, "std"), axis=1)
    desc = desc.drop((mape_col, "min"), axis=1)
    desc = desc.drop((mape_col, "mean"), axis=1)
    desc = desc.drop((mape_col, "max"), axis=1)
    order = ["No additional input", "Segmentation", "XNO2", "CSF"]

    desc_mape = desc.loc[order]

    df_mae_groupby = df_mae.groupby("Add. input:")
    desc = df_mae_groupby.describe()
    desc = desc.drop((mae_col, "count"), axis=1)
    desc = desc.drop((mae_col, "std"), axis=1)
    desc = desc.drop((mae_col, "min"), axis=1)
    desc = desc.drop((mae_col, "mean"), axis=1)
    desc = desc.drop((mae_col, "max"), axis=1)
    order = ["No additional input", "Segmentation", "XNO2", "CSF"]

    desc_mae = desc.loc[order]

    result = desc_mape.join(desc_mae)
    return {"res": result, "df_mae": df_mae, "df_mape": df_mape}


def integrated_gradients(model, img_tensor, baseline_tensor, num_steps=100):
    # Define the path from baseline to input as a straight line
    alphas = tf.linspace(start=0.0, stop=1.0, num=num_steps + 1)

    # Compute the gradients of the model's output with respect to the input
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
    grads = tape.gradient(predictions, img_tensor)

    # Compute the gradient at each point along the path
    interpolated_inputs = [
        (baseline_tensor + alpha * (img_tensor - baseline_tensor)) for alpha in alphas
    ]
    interpolated_inputs = tf.stack(interpolated_inputs)
    interpolated_inputs = tf.reshape(
        interpolated_inputs, [-1] + list(img_tensor.shape[1:])
    )
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        interpolated_predictions = model(interpolated_inputs)
    interpolated_grads = tape.gradient(
        interpolated_predictions, interpolated_inputs)

    # Approximate the integral using the trapezoidal rule
    avg_grads = tf.reduce_mean(interpolated_grads, axis=0)
    integrated_grads = tf.reduce_sum(
        avg_grads * (img_tensor - baseline_tensor), axis=0)
    return integrated_grads


def get_histo_inversion(
    model: tf.keras.Model, data: Data_eval, dir_save: str = "None"
) -> None:
    """Get various histograms summing up the inversion results."""
    metrics = get_inv_metrics_model_on_data(model, data)
    df_mae = pd.DataFrame({"loss": metrics["mae"], "method": "CNN-LSTM"})
    df_mape = pd.DataFrame({"loss": metrics["mape"], "method": "CNN-LSTM"})

    N_rows = 1
    N_cols = 2
    mympf.setMatplotlibParam()
    plt.viridis()
    fig, axs = mympf.set_figure_axs(
        N_rows,
        N_cols,
        wratio=.6,
        hratio=1,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.3,
        pad_h_ext=0.3,
        pad_h_int=0.35,
    )

    sns.kdeplot(
        data=df_mae,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[0],
    )
    sns.kdeplot(
        data=df_mape,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[1],
    )
    titles = [
        "Mean absolute error",
        "Mean absolute percentage error",
    ]

    for i_ax, ax in enumerate(axs):
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_xlabel(titles[i_ax])

    if dir_save != "None":
        plt.savefig(os.path.join(dir_save, "summary_inv.png"))


def sample_from(self, logits):
    logits, indices = ops.top_k(logits, k=self.k, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)


def generate_emiss_estimation():
    test_dataset_path = "/Users/xiaoxubeii/Downloads/data_paper_inv_pp/boxberg/test_dataset.nc"
    # model_name = "w_last.keras"
    predictor_model_res_path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/transformer/emiss_trans_2024-06-18_03-04-27"
    data = get_data_for_inversion(predictor_model_res_path, test_dataset_path)
    predictor = get_inversion_model(predictor_model_res_path)
    cfg = OmegaConf.load(os.path.join(predictor_model_res_path, "config.yaml"))
    embedding_path = "/".join(cfg.model.embedding_path.split("/")[:-1])
    embedding_layer = get_inversion_model_from_weights(embedding_path)
    embedding_layer.patch_encoder.downstream = True
    # metrics = get_inv_metrics_model_on_data(model, data)
    # predictor.mae = embedding

    def do_embedding(inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        def _embedding(inputs):
            outputs = keras.layers.Resizing(
                cfg.model.image_size, cfg.model.image_size)(inputs)
            patch_layer = embedding_layer.patch_layer
            patch_encoder = embedding_layer.patch_encoder
            encoder = embedding_layer.encoder

            patches = patch_layer(outputs)
            unmasked_embeddings = patch_encoder(patches)
            # Pass the unmaksed patch to the encoder.
            return encoder(unmasked_embeddings)

        embedding = tf.map_fn(_embedding, inputs)
        positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
        embedding = embedding + positional_encoding
        embedding_shape = embedding.shape
        return tf.reshape(
            embedding, [batch_size, seq_len, embedding_shape[-1]*embedding_shape[-2]])

    max_tokens = 1
    x = tf.convert_to_tensor(data.x.eval, np.float32)
    start_token = np.array([[x[0, 0]]])
    # start_token = predictor.embedding(start_token)
    start_token = do_embedding(start_token)
    num_tokens_generated = 0
    tokens_generated = []
    while num_tokens_generated < max_tokens:
        y = predictor.trans.predict(start_token, batch_size=1)
        tokens_generated.append(y)
        start_token = y
        num_tokens_generated = len(tokens_generated)

    tokens_generated = tf.squeeze(tokens_generated, axis=1)
    print(tokens_generated)
    emisses = predictor.predictor.predict(tokens_generated)
    emisses = np.squeeze(emisses)
    print(emisses)

    # pred = tf.convert_to_tensor(model.predict(x), np.float32)
    # y = tf.convert_to_tensor(data.y.eval, np.float32)[:10]

    # metrics = get_inv_metrics(y, pred)
    # print("mae:", np.mean(metrics["mae"]))
    # print("mape:", np.mean(metrics["mape"]))

    # plot_inversion_examples(
    #     data, metrics["mae"], metrics["mape"], model)
    # plot_inversion_examples(
    #     data, metrics["mae"], metrics["mape"], model)
    # get_summary_histo_inversion(model, data)
#     model_eval.get_histo_inversion(model, data)


if __name__ == "__main__":
    generate_emiss_estimation()
    # model_res_path = "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/mae/mae_2024-06-18_01-01-26"

    # cfg = OmegaConf.load(os.path.join(model_res_path, "config.yaml"))
    # model = Model_training_manager(cfg).model
    # model.load_weights(os.path.join(model_res_path, "w_last.weights.h5"))
    # import pdb
    # pdb.set_trace()

    # import h5py
    # f = h5py.File(
    #     "/Users/xiaoxubeii/Program/go/src/github.com/co2-images-inv-dl/res/mae/mae_2024-06-17_15-08-59/w_last.weights.h5", "r")

    # print(f.keys())
