"""
Description
This script defines a PyTorch Lightning module for PD L1 classification using only MONAI ViT. It supports 2D ViT and 3D ViT by changing spatial_dims, img_size, patch_size, and in_channels.

How to use
This file is imported by train_vit_pdl1.py. It is not usually run directly.
"""

import inspect
from pathlib import Path

import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics

from monai.networks.nets import ViT


def build_monai_vit(
    spatial_dims,
    in_channels,
    img_size,
    patch_size,
    num_classes,
    hidden_size,
    mlp_dim,
    num_layers,
    num_heads,
    dropout_rate,
    qkv_bias,
    pos_embed,
    proj_type,
    pos_embed_type,
    post_activation,
):
    signature = inspect.signature(ViT.__init__)
    available_parameters = signature.parameters

    kwargs = {
        "spatial_dims": spatial_dims,
        "in_channels": in_channels,
        "img_size": img_size,
        "patch_size": patch_size,
        "hidden_size": hidden_size,
        "mlp_dim": mlp_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "classification": True,
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
    }

    if "qkv_bias" in available_parameters:
        kwargs["qkv_bias"] = qkv_bias

    if "pos_embed" in available_parameters:
        kwargs["pos_embed"] = pos_embed

    if "proj_type" in available_parameters:
        kwargs["proj_type"] = proj_type

    if "pos_embed_type" in available_parameters:
        kwargs["pos_embed_type"] = pos_embed_type

    if "post_activation" in available_parameters:
        kwargs["post_activation"] = post_activation

    return ViT(**kwargs)


class PDL1ViTClassifier(pl.LightningModule):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        img_size,
        patch_size,
        num_classes,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.1,
        qkv_bias=False,
        pos_embed="conv",
        proj_type="conv",
        pos_embed_type="learnable",
        post_activation="none",
        learning_rate=1e-4,
        optimizer_name="adamw",
        weight_decay=1e-5,
        cosine_t_max=100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_monai_vit(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            pos_embed=pos_embed,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            post_activation=post_activation,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.train_metrics = self.create_metrics("train")
        self.val_metrics = self.create_metrics("val")
        self.test_metrics = self.create_metrics("test")

    def create_metrics(self, prefix):
        return torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task="multiclass",
                    average="weighted",
                    num_classes=self.hparams.num_classes,
                ),
                "auc": torchmetrics.classification.AUROC(
                    task="multiclass",
                    average="weighted",
                    num_classes=self.hparams.num_classes,
                ),
                "precision": torchmetrics.classification.Precision(
                    task="multiclass",
                    average="weighted",
                    num_classes=self.hparams.num_classes,
                ),
                "recall": torchmetrics.classification.Recall(
                    task="multiclass",
                    average="weighted",
                    num_classes=self.hparams.num_classes,
                ),
                "f1_score": torchmetrics.classification.F1Score(
                    task="multiclass",
                    average="weighted",
                    num_classes=self.hparams.num_classes,
                ),
            },
            prefix=f"{prefix}_",
        )

    def forward(self, images):
        output = self.model(images)

        if isinstance(output, tuple) or isinstance(output, list):
            return output[0]

        return output

    def training_step(self, batch, batch_idx):
        loss, probabilities, labels = self.common_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.shape[0])
        self.log_dict(self.train_metrics(probabilities, labels), on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probabilities, labels = self.common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.shape[0])
        self.log_dict(self.val_metrics(probabilities, labels), on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        loss, probabilities, labels = self.common_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.shape[0])
        self.log_dict(self.test_metrics(probabilities, labels), on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.shape[0])
        return loss

    def predict_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"].long()
        filenames = batch["image_meta_dict"]["filename_or_obj"]
        logits = self.forward(images)
        probabilities = torch.softmax(logits, dim=1)
        ids = [Path(filename).name.split("_")[0] for filename in filenames]

        return {
            "ids": ids,
            "labels": labels.detach().cpu(),
            "logits": logits.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
        }

    def common_step(self, batch):
        images = batch["image"]
        labels = batch["label"].long()
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)

        return loss, probabilities, labels

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_name.lower()

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_name: {self.hparams.optimizer_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.cosine_t_max,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
            },
        }
