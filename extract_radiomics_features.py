"""
Description
This script extracts PyRadiomics features from NIfTI image and mask pairs. It supports the original mask folder and multiple expanded or modified mask folders named like msk_00, msk_05, msk_10, ..., msk_100.

The expected input structure is:

data_root/
    CohortName1/
        img/
            image files ending with .nii.gz
        msk/
            original mask files ending with .nii.gz
    CohortName2/
        img/
        msk/
    RadiousMasks/
        msk_05/
            mask files ending with .nii.gz
        msk_10/
            mask files ending with .nii.gz
        ...
        msk_100/
            mask files ending with .nii.gz

The msk_00 output is created from the original masks inside each cohort folder. Other mask folders are read from RadiousMasks.

How to use
Install the required packages:

pip install pyradiomics SimpleITK pandas numpy tqdm pyyaml

Run the script:

python extract_radiomics_features.py \
    --data-root /path/to/AllCohort \
    --params-file /path/to/radiomics_params.yaml \
    --output-dir /path/to/save/RadiomicsFeatures/AllCohort

Optional example:

python extract_radiomics_features.py \
    --data-root ./AllCohort \
    --params-file ./radiomics_params.yaml \
    --output-dir ./RadiomicsFeatures/AllCohort \
    --radius-mask-dir RadiousMasks
"""

import argparse
import logging
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm


warnings.filterwarnings("ignore")
radiomics.setVerbosity(logging.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features from NIfTI images and masks.")

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--params-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--radius-mask-dir", type=str, default="RadiousMasks")
    parser.add_argument("--radius-start", type=int, default=0)
    parser.add_argument("--radius-stop", type=int, default=100)
    parser.add_argument("--radius-step", type=int, default=5)
    parser.add_argument("--radius-prefix", type=str, default="msk_")

    parser.add_argument("--timezone", type=str, default="America/Chicago")
    parser.add_argument("--log-file", type=Path, default=None)

    return parser.parse_args()


def configure_logging(log_file=None):
    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def validate_inputs(data_root, params_file):
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if not params_file.exists():
        raise FileNotFoundError(f"Parameter file does not exist: {params_file}")


def remove_nii_suffix(path):
    name = Path(path).name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return Path(name).stem


def get_subject_id(path):
    stem = remove_nii_suffix(path)
    match = re.search(r"\d+", stem)

    if match is not None:
        return match.group(0)

    return stem


def get_radius_names(start, stop, step, prefix):
    return [f"{prefix}{value:02d}" for value in range(start, stop + 1, step)]


def find_image_paths(data_root):
    return sorted(data_root.glob("*/img/*.nii.gz"))


def find_original_mask_paths(data_root):
    return sorted(data_root.glob("*/msk/*.nii.gz"))


def find_radius_mask_paths(data_root, radius_mask_dir, radius_name):
    return sorted((data_root / radius_mask_dir / radius_name).glob("*.nii.gz"))


def pair_images_and_masks(image_paths, mask_paths):
    masks_by_id = defaultdict(list)

    for mask_path in mask_paths:
        masks_by_id[get_subject_id(mask_path)].append(mask_path)

    pairs = []
    missing_images = []
    duplicated_mask_ids = []

    for subject_id, paths in masks_by_id.items():
        if len(paths) > 1:
            duplicated_mask_ids.append(subject_id)

    for image_path in image_paths:
        subject_id = get_subject_id(image_path)
        matched_masks = masks_by_id.get(subject_id, [])

        if len(matched_masks) == 0:
            missing_images.append(image_path)
            continue

        pairs.append(
            {
                "image": image_path,
                "label": matched_masks[0],
            }
        )

    if len(pairs) == 0 and len(image_paths) == len(mask_paths):
        pairs = [
            {
                "image": image_path,
                "label": mask_path,
            }
            for image_path, mask_path in zip(sorted(image_paths), sorted(mask_paths))
        ]

    return pairs, missing_images, duplicated_mask_ids


def read_sitk_image(file_path):
    return sitk.ReadImage(str(file_path), sitk.sitkFloat64)


def convert_value(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value

    return value


def extract_radiomics_features(image_path, mask_path, extractor):
    image = read_sitk_image(image_path)
    mask = read_sitk_image(mask_path) > 0

    features = extractor.execute(image, mask, voxelBased=False)
    feature_names = sorted([name for name in features if not name.startswith("diagnostics_")])
    feature_values = [convert_value(features[name]) for name in feature_names]

    subject_name = remove_nii_suffix(image_path)

    return pd.DataFrame(
        data=[feature_values],
        columns=feature_names,
        index=[subject_name],
    )


def build_dataset_by_radius(data_root, radius_mask_dir, radius_names):
    image_paths = find_image_paths(data_root)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files were found under: {data_root}")

    dataset_by_radius = {}

    for radius_name in radius_names:
        if radius_name == "msk_00":
            mask_paths = find_original_mask_paths(data_root)
        else:
            mask_paths = find_radius_mask_paths(data_root, radius_mask_dir, radius_name)

        pairs, missing_images, duplicated_mask_ids = pair_images_and_masks(image_paths, mask_paths)

        if len(mask_paths) == 0:
            logging.warning(f"No masks found for {radius_name}")

        if len(missing_images) > 0:
            logging.warning(f"{radius_name}: {len(missing_images)} images did not have matched masks")

        if len(duplicated_mask_ids) > 0:
            logging.warning(f"{radius_name}: duplicated mask IDs found: {duplicated_mask_ids}")

        dataset_by_radius[radius_name] = pairs

    return dataset_by_radius


def save_problematic_cases(problematic_cases, output_dir):
    if len(problematic_cases) == 0:
        return

    problematic_df = pd.DataFrame(problematic_cases)
    problematic_df.to_csv(output_dir / "problematic_cases.csv", index=False)


def run_extraction(args):
    validate_inputs(args.data_root, args.params_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    radius_names = get_radius_names(
        start=args.radius_start,
        stop=args.radius_stop,
        step=args.radius_step,
        prefix=args.radius_prefix,
    )

    dataset_by_radius = build_dataset_by_radius(
        data_root=args.data_root,
        radius_mask_dir=args.radius_mask_dir,
        radius_names=radius_names,
    )

    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.params_file))
    problematic_cases = []

    logging.info("Cohort list has been created")

    for radius_name, cohort in dataset_by_radius.items():
        if len(cohort) == 0:
            logging.warning(f"{radius_name}: no valid image and mask pairs found")
            continue

        current_time = datetime.now(pytz.timezone(args.timezone)).strftime("%m/%d/%Y, %H:%M:%S")
        start_time = time.time()
        extracted_features = []

        logging.info(f"@ {current_time} extracting radiomics features for {radius_name}")

        for subject_paths in tqdm(cohort, desc=radius_name):
            image_path = subject_paths["image"]
            mask_path = subject_paths["label"]

            try:
                extracted_features.append(
                    extract_radiomics_features(
                        image_path=image_path,
                        mask_path=mask_path,
                        extractor=extractor,
                    )
                )
            except Exception as error:
                problematic_cases.append(
                    {
                        "radius": radius_name,
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "error": str(error),
                    }
                )

        if len(extracted_features) == 0:
            logging.warning(f"{radius_name}: no features were extracted")
            continue

        final_df = pd.concat(extracted_features, axis=0)
        output_path = args.output_dir / f"{radius_name}.csv"
        final_df.to_csv(output_path)

        elapsed_time = time.time() - start_time
        logging.info(f"Saved: {output_path}")
        logging.info(f"Elapsed time for {radius_name}: {elapsed_time:.1f} seconds")

    save_problematic_cases(problematic_cases, args.output_dir)

    if len(problematic_cases) > 0:
        logging.warning(f"Finished with {len(problematic_cases)} problematic cases")
    else:
        logging.info("Finished without problematic cases")


def main():
    args = parse_args()
    configure_logging(args.log_file)
    run_extraction(args)


if __name__ == "__main__":
    main()