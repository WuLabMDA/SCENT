"""
Description
This script trains a MONAI ViT model for PD L1 classification using cropped 2D or 3D tumor patches.

The script has three ViT modes.

2d_rgb
Use this for 2D patches or 3 slice patches. The model is a 2D ViT with 3 input channels.

2d_channels
Use this for 3D patches when the full depth is used as channels. For example, a 32 slice patch becomes a 2D ViT input with 32 channels.

3d
Use this for true 3D ViT. For example, a 32, 224, 224 input is converted to 1, 32, 192, 192 by default and trained with patch size 16, 96, 96.

How to use
Example for 2D RGB ViT:

python train_vit_pdl1.py --data-dir /path/to/DeepLearningBased/AllCohort/2D/tissue_pad_32 --metadata-file /path/to/Theranostic_surrogate_marker_.xlsx --output-root /path/to/outputs --vit-mode 2d_rgb --target-mode twoclass

Example for 3D patches as 2D channels:

python train_vit_pdl1.py --data-dir /path/to/DeepLearningBased/AllCohort/3D_32_224_224/tissue_pad_32 --metadata-file /path/to/Theranostic_surrogate_marker_.xlsx --output-root /path/to/outputs --vit-mode 2d_channels --input-depth 32 --target-mode twoclass

Example for true 3D ViT:

python train_vit_pdl1.py --data-dir /path/to/DeepLearningBased/AllCohort/3D_32_224_224/tissue_pad_32 --metadata-file /path/to/Theranostic_surrogate_marker_.xlsx --output-root /path/to/outputs --vit-mode 3d --input-depth 32 --spatial-size 192 --patch-size-3d 16 96 96 --target-mode twoclass
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner
from monai.utils import set_determinism

from pdl1_datamodule import PDL1DataModule
from vit_lightning_module import PDL1ViTClassifier


torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MONAI ViT for PD L1 classification.")

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)

    parser.add_argument("--target-mode", type=str, default="twoclass", choices=["twoclass", "threeclass"])
    parser.add_argument("--two-class-threshold", type=float, default=1.0)
    parser.add_argument("--three-class-thresholds", type=float, nargs=2, default=[1.0, 49.0])

    parser.add_argument("--vit-mode", type=str, default="auto", choices=["auto", "2d_rgb", "2d_channels", "3d"])
    parser.add_argument("--volume-strategy", type=str, default="channels2d", choices=["channels2d", "volume3d"])
    parser.add_argument("--input-depth", type=int, default=None)
    parser.add_argument("--spatial-size", type=int, default=None)
    parser.add_argument("--axis-order", type=str, default="hwd", choices=["hwd", "dhw"])

    parser.add_argument("--patch-size-2d", type=int, default=16)
    parser.add_argument("--patch-size-3d", type=int, nargs=3, default=[16, 96, 96])

    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--mlp-dim", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")
    parser.add_argument("--pos-embed", type=str, default="conv")
    parser.add_argument("--proj-type", type=str, default="conv")
    parser.add_argument("--pos-embed-type", type=str, default="learnable")
    parser.add_argument("--post-activation", type=str, default="none")

    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.5)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-name", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--cosine-t-max", type=int, default=100)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--early-stopping-monitor", type=str, default="val_loss")
    parser.add_argument("--early-stopping-mode", type=str, default="min")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint-monitor", type=str, default="val_accuracy")
    parser.add_argument("--checkpoint-mode", type=str, default="max")

    parser.add_argument("--run-lr-finder", action="store_true")

    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_devices(devices):
    if devices == "auto":
        return "auto"

    if "," in devices:
        return [int(value.strip()) for value in devices.split(",") if value.strip() != ""]

    if devices.isdigit():
        return int(devices)

    return devices


def infer_depth_from_path(data_dir):
    path_text = str(data_dir)
    match = re.search(r"3D_(\d+)_(\d+)_(\d+)", path_text)

    if match is not None:
        return int(match.group(1)), int(match.group(2))

    if re.search(r"(^|/)2D(/|$)", path_text):
        return 3, 224

    return None, None


def resolve_vit_mode(args):
    if args.vit_mode != "auto":
        return args.vit_mode

    inferred_depth, _ = infer_depth_from_path(args.data_dir)

    if inferred_depth == 3:
        return "2d_rgb"

    if args.volume_strategy == "volume3d":
        return "3d"

    return "2d_channels"


def resolve_input_depth(args, vit_mode):
    if args.input_depth is not None:
        return args.input_depth

    inferred_depth, _ = infer_depth_from_path(args.data_dir)

    if inferred_depth is not None:
        return inferred_depth

    if vit_mode == "2d_rgb":
        return 3

    raise ValueError("input_depth could not be inferred. Please provide --input-depth.")


def resolve_spatial_size(args, vit_mode):
    if args.spatial_size is not None:
        return args.spatial_size

    _, inferred_spatial_size = infer_depth_from_path(args.data_dir)

    if vit_mode == "3d":
        return 192

    if inferred_spatial_size is not None:
        return inferred_spatial_size

    return 224


def get_num_classes(target_mode):
    if target_mode == "twoclass":
        return 2

    if target_mode == "threeclass":
        return 3

    raise ValueError(f"Unsupported target_mode: {target_mode}")


def get_vit_geometry(vit_mode, input_depth, spatial_size, patch_size_2d, patch_size_3d):
    if vit_mode == "3d":
        return {
            "spatial_dims": 3,
            "in_channels": 1,
            "img_size": (input_depth, spatial_size, spatial_size),
            "patch_size": tuple(patch_size_3d),
        }

    if vit_mode == "2d_rgb":
        return {
            "spatial_dims": 2,
            "in_channels": 3,
            "img_size": (spatial_size, spatial_size),
            "patch_size": (patch_size_2d, patch_size_2d),
        }

    if vit_mode == "2d_channels":
        return {
            "spatial_dims": 2,
            "in_channels": input_depth,
            "img_size": (spatial_size, spatial_size),
            "patch_size": (patch_size_2d, patch_size_2d),
        }

    raise ValueError(f"Unsupported vit_mode: {vit_mode}")


def validate_vit_geometry(img_size, patch_size):
    for image_dim, patch_dim in zip(img_size, patch_size):
        if image_dim % patch_dim != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")


def make_experiment_name(args, vit_mode, input_depth, spatial_size):
    if args.experiment_name is not None:
        return args.experiment_name

    data_name = "_".join(args.data_dir.parts[-4:])
    data_name = re.sub(r"[^A-Za-z0-9]+", "_", data_name).strip("_")

    return f"vit_{vit_mode}_{args.target_mode}_D{input_depth}_S{spatial_size}_{data_name}"


def create_trainer(args, experiment_name):
    checkpoint_dir = args.output_root / "checkpoints" / experiment_name
    logger = CSVLogger(save_dir=str(args.output_root / "lightning_logs"), name=experiment_name)

    early_stopping = EarlyStopping(
        monitor=args.early_stopping_monitor,
        min_delta=0.0,
        patience=args.patience,
        verbose=False,
        mode=args.early_stopping_mode,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=experiment_name,
        save_top_k=1,
        monitor=args.checkpoint_monitor,
        verbose=True,
        mode=args.checkpoint_mode,
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    return pl.Trainer(
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[early_stopping, checkpoint_callback, learning_rate_monitor],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
    )


def save_metric_plot(metrics, columns, xlabel, ylabel, output_path):
    available_columns = [column for column in columns if column in metrics.columns]

    if len(available_columns) == 0:
        return

    ax = metrics[available_columns].plot(grid=True, legend=True, xlabel=xlabel, ylabel=ylabel)
    fig = ax.get_figure()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_plots(log_dir):
    metrics_path = Path(log_dir) / "metrics.csv"

    if not metrics_path.exists():
        return

    metrics = pd.read_csv(metrics_path)

    if "epoch" in metrics.columns:
        epoch_metrics = []
        for epoch, group in metrics.groupby("epoch"):
            values = group.mean(numeric_only=True).to_dict()
            values["epoch"] = epoch
            epoch_metrics.append(values)

        epoch_metrics = pd.DataFrame(epoch_metrics)

        save_metric_plot(epoch_metrics, ["train_loss", "val_loss"], "Epoch", "Loss", Path(log_dir) / "loss.png")
        save_metric_plot(epoch_metrics, ["train_auc", "val_auc"], "Epoch", "AUC", Path(log_dir) / "auc.png")
        save_metric_plot(epoch_metrics, ["train_accuracy", "val_accuracy"], "Epoch", "Accuracy", Path(log_dir) / "accuracy.png")

    if "step" in metrics.columns:
        step_metrics = []
        for step, group in metrics.groupby("step"):
            values = group.mean(numeric_only=True).to_dict()
            values["step"] = step
            step_metrics.append(values)

        step_metrics = pd.DataFrame(step_metrics)
        lr_columns = [column for column in step_metrics.columns if column.startswith("lr-")]
        save_metric_plot(step_metrics, lr_columns, "Step", "Learning Rate", Path(log_dir) / "learning_rate.png")


def save_predictions(predictions, data_module, output_path, num_classes):
    train_ids = set(item["idx"] for item in data_module.train_dict)
    val_ids = set(item["idx"] for item in data_module.val_dict)

    rows = []

    for batch in predictions:
        ids = batch["ids"]
        labels = batch["labels"].numpy()
        probabilities = batch["probabilities"].numpy()
        logits = batch["logits"].numpy()
        predicted_labels = np.argmax(probabilities, axis=1)

        for row_index, subject_id in enumerate(ids):
            if subject_id in train_ids:
                cohort = "Train"
            elif subject_id in val_ids:
                cohort = "Validation"
            else:
                cohort = "Unknown"

            row = {
                "idx": subject_id,
                "label": int(labels[row_index]),
                "pred": int(predicted_labels[row_index]),
                "cohort": cohort,
            }

            for class_index in range(num_classes):
                row[f"prob_class_{class_index}"] = float(probabilities[row_index, class_index])
                row[f"logit_class_{class_index}"] = float(logits[row_index, class_index])

            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    args = parse_args()
    configure_logging()
    set_determinism(seed=args.seed)

    vit_mode = resolve_vit_mode(args)
    input_depth = resolve_input_depth(args, vit_mode)
    spatial_size = resolve_spatial_size(args, vit_mode)
    num_classes = get_num_classes(args.target_mode)

    geometry = get_vit_geometry(
        vit_mode=vit_mode,
        input_depth=input_depth,
        spatial_size=spatial_size,
        patch_size_2d=args.patch_size_2d,
        patch_size_3d=args.patch_size_3d,
    )

    validate_vit_geometry(geometry["img_size"], geometry["patch_size"])

    experiment_name = make_experiment_name(args, vit_mode, input_depth, spatial_size)

    logging.info(f"Experiment name: {experiment_name}")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"ViT mode: {vit_mode}")
    logging.info(f"Input depth: {input_depth}")
    logging.info(f"Spatial size: {spatial_size}")
    logging.info(f"Image size: {geometry['img_size']}")
    logging.info(f"Patch size: {geometry['patch_size']}")

    data_module = PDL1DataModule(
        data_dir=args.data_dir,
        meta_path=args.metadata_file,
        target_mode=args.target_mode,
        threshold_twoclass=args.two_class_threshold,
        threshold_threeclass=tuple(args.three_class_thresholds),
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
        vit_mode=vit_mode,
        input_depth=input_depth,
        spatial_size=spatial_size,
        axis_order=args.axis_order,
    )

    data_module.prepare_data()
    data_module.setup()

    logging.info(f"Training cases: {len(data_module.train_ds)}")
    logging.info(f"Validation cases: {len(data_module.val_ds)}")
    logging.info(f"Test cases: {len(data_module.test_ds)}")
    logging.info(f"Prediction cases: {len(data_module.predict_ds)}")

    model = PDL1ViTClassifier(
        spatial_dims=geometry["spatial_dims"],
        in_channels=geometry["in_channels"],
        img_size=geometry["img_size"],
        patch_size=geometry["patch_size"],
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        qkv_bias=args.qkv_bias,
        pos_embed=args.pos_embed,
        proj_type=args.proj_type,
        pos_embed_type=args.pos_embed_type,
        post_activation=args.post_activation,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer_name,
        weight_decay=args.weight_decay,
        cosine_t_max=args.cosine_t_max,
    )

    trainer = create_trainer(args, experiment_name)

    if args.run_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model=model, datamodule=data_module)
        suggested_lr = lr_finder.suggestion()

        if suggested_lr is not None:
            model.learning_rate = suggested_lr
            logging.info(f"Learning rate changed to: {suggested_lr}")

        figure = lr_finder.plot(suggest=True)
        lr_plot_path = Path(trainer.logger.log_dir) / "lr_finder.png"
        lr_plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(lr_plot_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    start_time = datetime.now()
    logging.info(f"Training started at: {start_time}")

    trainer.fit(model=model, datamodule=data_module)

    logging.info(f"Training duration: {datetime.now() - start_time}")

    save_training_plots(trainer.logger.log_dir)

    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    logging.info(f"Best checkpoint path: {best_checkpoint_path}")

    best_model = PDL1ViTClassifier.load_from_checkpoint(best_checkpoint_path)
    best_model.eval()

    trainer.validate(model=best_model, datamodule=data_module)
    trainer.test(model=best_model, datamodule=data_module)

    predictions = trainer.predict(model=best_model, datamodule=data_module)

    prediction_path = args.output_root / "predictions" / f"{experiment_name}_predictions.csv"
    save_predictions(predictions, data_module, prediction_path, num_classes)

    logging.info(f"Predictions saved to: {prediction_path}")
    logging.info("Finished")


if __name__ == "__main__":
    main()
