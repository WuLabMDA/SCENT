"""
Description
This script defines the Lightning data module used for PD L1 classification from 2D or 3D NIfTI tumor patches. It reads cropped image patches, matches them with the clinical metadata Excel file, creates two class or three class PD L1 labels, applies MONAI transforms, and prepares train, validation, test, and prediction dataloaders.

The script supports three input modes. In 2d_rgb mode, the input becomes C, H, W with three channels. In 2d_channels mode, the full depth becomes channels, for example 32, H, W. In 3d mode, the input becomes C, D, H, W with one channel.

How to use
This file is imported by train_vit_pdl1.py. It is not usually run directly.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from monai.data import CacheDataset, DataLoader, partition_dataset_classes
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    EnsureTyped,
)


class PrepareViTInputd(MapTransform):
    def __init__(self, keys, vit_mode, input_depth=None, axis_order="hwd"):
        super().__init__(keys)
        self.vit_mode = vit_mode
        self.input_depth = input_depth
        self.axis_order = axis_order

    def __call__(self, data):
        output = dict(data)

        for key in self.keys:
            image = torch.as_tensor(output[key]).float()
            volume = self.to_dhw(image)

            if self.vit_mode == "2d_rgb":
                output[key] = self.to_2d_rgb(volume)
            elif self.vit_mode == "2d_channels":
                output[key] = self.to_2d_channels(volume)
            elif self.vit_mode == "3d":
                output[key] = self.to_3d(volume)
            else:
                raise ValueError(f"Unsupported vit_mode: {self.vit_mode}")

        return output

    def to_dhw(self, image):
        while image.ndim > 3 and image.shape[0] == 1:
            image = image.squeeze(0)

        image = image.squeeze()

        if image.ndim == 2:
            image = image.unsqueeze(-1)

        if image.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image, but got shape {tuple(image.shape)}")

        if self.axis_order == "hwd":
            return image.permute(2, 0, 1).contiguous()

        if self.axis_order == "dhw":
            return image.contiguous()

        raise ValueError(f"Unsupported axis_order: {self.axis_order}")

    def fit_depth(self, volume):
        if self.input_depth is None:
            return volume

        current_depth = volume.shape[0]

        if current_depth == self.input_depth:
            return volume

        if current_depth > self.input_depth:
            start = (current_depth - self.input_depth) // 2
            return volume[start : start + self.input_depth]

        fill_value = float(torch.min(volume))
        output = torch.full(
            size=(self.input_depth, volume.shape[1], volume.shape[2]),
            fill_value=fill_value,
            dtype=volume.dtype,
        )
        start = (self.input_depth - current_depth) // 2
        output[start : start + current_depth] = volume

        return output

    def to_2d_rgb(self, volume):
        volume = self.fit_depth(volume)

        if volume.shape[0] == 3:
            return volume

        center_slice = volume[volume.shape[0] // 2]
        return center_slice.unsqueeze(0).repeat(3, 1, 1)

    def to_2d_channels(self, volume):
        return self.fit_depth(volume)

    def to_3d(self, volume):
        volume = self.fit_depth(volume)
        return volume.unsqueeze(0)


class PDL1DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        meta_path,
        target_mode="twoclass",
        threshold_twoclass=1.0,
        threshold_threeclass=(1.0, 49.0),
        train_batch_size=16,
        val_batch_size=16,
        test_batch_size=16,
        num_workers=8,
        train_ratio=0.5,
        seed=42,
        vit_mode="2d_rgb",
        input_depth=None,
        spatial_size=224,
        axis_order="hwd",
        id_column="NewID",
        site_column="Site_cat",
        pdl1_column="PD_L1Expression",
        pfs_status_column="PFSStatus",
        pfs_column="PFS",
        os_status_column="OSStatus",
        os_column="OS",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.meta_path = Path(meta_path)
        self.target_mode = target_mode
        self.threshold_twoclass = threshold_twoclass
        self.threshold_threeclass = threshold_threeclass
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.seed = seed
        self.vit_mode = vit_mode
        self.input_depth = input_depth
        self.spatial_size = spatial_size
        self.axis_order = axis_order
        self.id_column = id_column
        self.site_column = site_column
        self.pdl1_column = pdl1_column
        self.pfs_status_column = pfs_status_column
        self.pfs_column = pfs_column
        self.os_status_column = os_status_column
        self.os_column = os_column

    def two_class(self, value):
        if pd.isna(value):
            return None

        if float(value) > self.threshold_twoclass:
            return 1

        return 0

    def three_class(self, value):
        if pd.isna(value):
            return None

        value = float(value)
        lower_threshold = float(self.threshold_threeclass[0])
        upper_threshold = float(self.threshold_threeclass[1])

        if value > upper_threshold:
            return 2

        if value > lower_threshold:
            return 1

        return 0

    def prepare_data(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {self.meta_path}")

        image_paths = sorted(self.data_dir.rglob("*.nii.gz"))
        metadata = self.load_metadata()

        if len(image_paths) == 0:
            raise FileNotFoundError(f"No .nii.gz files were found in: {self.data_dir}")

        self.data_dicts = []
        skipped_cases = []

        for image_path in image_paths:
            subject_id = self.get_subject_id(image_path)
            metadata_row = metadata[metadata["_match_id"] == subject_id]

            if len(metadata_row) == 0:
                skipped_cases.append((str(image_path), "Missing metadata"))
                continue

            row = metadata_row.iloc[0]
            label = self.get_label(row[self.pdl1_column])

            if label is None:
                skipped_cases.append((str(image_path), "Missing target label"))
                continue

            self.data_dicts.append(
                {
                    "idx": subject_id,
                    "image": str(image_path),
                    "Site": row[self.site_column],
                    "PDL1": row[self.pdl1_column],
                    "PFSStatus": row[self.pfs_status_column],
                    "PFS": row[self.pfs_column],
                    "OSStatus": row[self.os_status_column],
                    "OS": row[self.os_column],
                    "twoclass": self.two_class(row[self.pdl1_column]),
                    "threeclass": self.three_class(row[self.pdl1_column]),
                    "label": int(label),
                }
            )

        if len(self.data_dicts) == 0:
            raise ValueError("No valid cases were found after metadata matching")

        if len(skipped_cases) > 0:
            logging.warning(f"Skipped {len(skipped_cases)} cases because of missing metadata or labels")

    def load_metadata(self):
        metadata = pd.read_excel(self.meta_path)
        metadata["_match_id"] = metadata[self.id_column].apply(self.format_metadata_id)
        return metadata

    def format_metadata_id(self, value):
        if pd.isna(value):
            return ""

        return str(int(value)).zfill(4)

    def get_subject_id(self, image_path):
        name = Path(image_path).name

        if name.endswith(".nii.gz"):
            name = name[:-7]
        elif name.endswith(".nii"):
            name = name[:-4]

        matches = re.findall(r"\d+", name)

        if len(matches) == 0:
            return name

        return matches[0].zfill(4)

    def get_label(self, value):
        if self.target_mode == "twoclass":
            return self.two_class(value)

        if self.target_mode == "threeclass":
            return self.three_class(value)

        raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def train_transforms(self):
        keys = ["image"]
        transforms = [
            LoadImaged(keys=keys, image_only=False),
            PrepareViTInputd(
                keys=keys,
                vit_mode=self.vit_mode,
                input_depth=self.input_depth,
                axis_order=self.axis_order,
            ),
            self.get_resize_transform(keys),
            ScaleIntensityd(keys=keys),
        ]

        if self.vit_mode == "3d":
            transforms.extend(
                [
                    RandRotated(
                        keys=keys,
                        range_x=np.pi / 18,
                        range_y=np.pi / 18,
                        range_z=np.pi / 18,
                        prob=0.3,
                        keep_size=True,
                    ),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                ]
            )
        else:
            transforms.extend(
                [
                    RandRotated(keys=keys, range_x=np.pi / 12, prob=0.5, keep_size=True),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                ]
            )

        transforms.extend(
            [
                RandShiftIntensityd(keys=keys, offsets=0.10, prob=0.50),
                RandGaussianNoised(keys=keys, prob=0.25),
                NormalizeIntensityd(keys=keys, nonzero=False, channel_wise=True),
                EnsureTyped(keys=keys),
            ]
        )

        return Compose(transforms)

    def val_transforms(self):
        keys = ["image"]

        return Compose(
            [
                LoadImaged(keys=keys, image_only=False),
                PrepareViTInputd(
                    keys=keys,
                    vit_mode=self.vit_mode,
                    input_depth=self.input_depth,
                    axis_order=self.axis_order,
                ),
                self.get_resize_transform(keys),
                ScaleIntensityd(keys=keys),
                NormalizeIntensityd(keys=keys, nonzero=False, channel_wise=True),
                EnsureTyped(keys=keys),
            ]
        )

    def get_resize_transform(self, keys):
        if self.vit_mode == "3d":
            spatial_size = (self.input_depth, self.spatial_size, self.spatial_size)
        else:
            spatial_size = (self.spatial_size, self.spatial_size)

        return ResizeWithPadOrCropd(keys=keys, spatial_size=spatial_size)

    def setup(self, stage=None):
        split_data = partition_dataset_classes(
            self.data_dicts,
            classes=[item["label"] for item in self.data_dicts],
            ratios=[self.train_ratio, 1.0 - self.train_ratio],
            shuffle=True,
            seed=self.seed,
        )

        self.train_dict = split_data[0]
        self.val_dict = split_data[1]
        self.test_dict = split_data[1]

        self.train_ds = CacheDataset(
            data=self.train_dict,
            transform=self.train_transforms(),
            num_workers=self.num_workers,
            copy_cache=False,
        )

        self.val_ds = CacheDataset(
            data=self.val_dict,
            transform=self.val_transforms(),
            num_workers=self.num_workers,
            copy_cache=False,
        )

        self.test_ds = CacheDataset(
            data=self.test_dict,
            transform=self.val_transforms(),
            num_workers=self.num_workers,
            copy_cache=False,
        )

        self.predict_ds = CacheDataset(
            data=self.data_dicts,
            transform=self.val_transforms(),
            num_workers=self.num_workers,
            copy_cache=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
