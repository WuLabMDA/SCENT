"""
Description
This script creates a summary table for a cohort of CT images and tumor masks stored as NIfTI files. It matches each image and mask using the subject ID in the file name, connects the imaging data with a clinical metadata Excel file, and saves a summary table with patient information, PD L1 class labels, mask bounding box size, image intensity values inside the tumor region, image shape, spacing, and lesion number.

The expected input image structure is:

data_root/
    CohortName1/
        img/
            img0001.nii.gz
        msk/
            mask0001.nii.gz
    CohortName2/
        img/
        msk/

The metadata Excel file should contain one row per patient and must include the patient ID column and the clinical columns used in this script. By default, the patient ID column is NewID and it will be converted to a 4 digit ID for matching.

How to use
Install the required packages:

pip install SimpleITK pandas numpy scikit-image tqdm openpyxl

Run the script:

python create_data_summary.py \
    --data-root /path/to/RawData/AllCohort \
    --metadata-file /path/to/Theranostic_surrogate_marker_.xlsx \
    --output-file /path/to/DataSummary_Label_v2.xlsx

Optional example:

python create_data_summary.py \
    --data-root ./AllCohort \
    --metadata-file ./metadata.xlsx \
    --output-file ./outputs/DataSummary_Label_v2.xlsx \
    --id-column NewID
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label
from tqdm import tqdm


SUMMARY_COLUMNS = [
    "id",
    "Site",
    "PD_L1Expression",
    "PFSStatus",
    "PFS",
    "OSStatus",
    "OS",
    "twoclass",
    "threeclass",
    "Labels Value",
    "Lesion Number",
    "Min Label slice",
    "Max Label slice",
    "Region Depth",
    "Min Label Height",
    "Max Label Height",
    "Region Height",
    "Min Label Width",
    "Max Label Width",
    "Region Width",
    "Region Size",
    "img.shape==msk.shape",
    "img.shape",
    "Min_img",
    "Max_img",
    "Avg_img",
    "Equal Spacing",
    "spacing[0]",
    "spacing[1]",
    "spacing[2]",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create data summary table for NIfTI images and masks.")

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)

    parser.add_argument("--image-glob", type=str, default="*/img/*.nii.gz")
    parser.add_argument("--mask-glob", type=str, default="*/msk/*.nii.gz")

    parser.add_argument("--id-column", type=str, default="NewID")
    parser.add_argument("--site-column", type=str, default="Site_cat")
    parser.add_argument("--pdl1-column", type=str, default="PD_L1Expression")
    parser.add_argument("--pfs-status-column", type=str, default="PFSStatus")
    parser.add_argument("--pfs-column", type=str, default="PFS")
    parser.add_argument("--os-status-column", type=str, default="OSStatus")
    parser.add_argument("--os-column", type=str, default="OS")

    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_inputs(data_root, metadata_file):
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_file}")


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

    if match is None:
        return stem

    return match.group(0).zfill(4)


def two_class(value):
    if pd.isna(value):
        return np.nan

    if value > 49:
        return 1

    return 0


def three_class(value):
    if pd.isna(value):
        return np.nan

    if value > 49:
        return 2

    if value > 0:
        return 1

    return 0


def read_image(image_path):
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(image)

    return array, image


def get_label_values(mask_array):
    return sorted(np.unique(mask_array).astype(float).tolist())


def get_label_boundary(mask_array):
    z_positions, y_positions, x_positions = np.where(mask_array > 0)

    if len(z_positions) == 0:
        return None

    return [
        int(np.min(z_positions)),
        int(np.max(z_positions)),
        int(np.min(y_positions)),
        int(np.max(y_positions)),
        int(np.min(x_positions)),
        int(np.max(x_positions)),
    ]


def count_connected_lesions(mask_array):
    connected_mask = label(mask_array > 0)

    return int(np.max(connected_mask))


def load_metadata(metadata_file, id_column):
    metadata = pd.read_excel(metadata_file)
    metadata["Cross_ID"] = metadata[id_column].apply(lambda value: str(int(value)).zfill(4))

    return metadata


def get_metadata_row(metadata, subject_id):
    matched_rows = metadata[metadata["Cross_ID"] == subject_id]

    if len(matched_rows) == 0:
        raise ValueError(f"No metadata row was found for subject ID: {subject_id}")

    if len(matched_rows) > 1:
        logging.warning(f"More than one metadata row was found for subject ID: {subject_id}. The first row was used.")

    return matched_rows.iloc[0]


def find_image_mask_pairs(data_root, image_glob, mask_glob):
    image_paths = sorted(data_root.glob(image_glob))
    mask_paths = sorted(data_root.glob(mask_glob))

    masks_by_id = {get_subject_id(mask_path): mask_path for mask_path in mask_paths}

    pairs = []
    missing_masks = []

    for image_path in image_paths:
        subject_id = get_subject_id(image_path)
        mask_path = masks_by_id.get(subject_id)

        if mask_path is None:
            missing_masks.append(str(image_path))
            continue

        pairs.append(
            {
                "id": subject_id,
                "image": image_path,
                "mask": mask_path,
            }
        )

    if len(missing_masks) > 0:
        logging.warning(f"{len(missing_masks)} images did not have matched masks")

    return pairs


def create_empty_mask_summary(subject_info):
    row = {column: np.nan for column in SUMMARY_COLUMNS}

    row.update(subject_info)
    row["Labels Value"] = []
    row["Lesion Number"] = 0

    return row


def analyze_sample(sample, metadata_row, args):
    image_array, image_header = read_image(sample["image"])
    mask_array, mask_header = read_image(sample["mask"])

    subject_info = {
        "id": sample["id"],
        "Site": metadata_row[args.site_column],
        "PD_L1Expression": metadata_row[args.pdl1_column],
        "PFSStatus": metadata_row[args.pfs_status_column],
        "PFS": metadata_row[args.pfs_column],
        "OSStatus": metadata_row[args.os_status_column],
        "OS": metadata_row[args.os_column],
        "twoclass": two_class(metadata_row[args.pdl1_column]),
        "threeclass": three_class(metadata_row[args.pdl1_column]),
    }

    label_values = get_label_values(mask_array)
    label_boundary = get_label_boundary(mask_array)

    if label_boundary is None:
        return create_empty_mask_summary(subject_info)

    z_min, z_max, y_min, y_max, x_min, x_max = label_boundary

    crop_image = image_array[
        z_min : z_max + 1,
        y_min : y_max + 1,
        x_min : x_max + 1,
    ]

    tumor_region = image_array[mask_array > 0]

    image_spacing = image_header.GetSpacing()
    mask_spacing = mask_header.GetSpacing()

    row = {
        **subject_info,
        "Labels Value": str(label_values),
        "Lesion Number": count_connected_lesions(mask_array),
        "Min Label slice": z_min,
        "Max Label slice": z_max,
        "Region Depth": crop_image.shape[0],
        "Min Label Height": y_min,
        "Max Label Height": y_max,
        "Region Height": crop_image.shape[1],
        "Min Label Width": x_min,
        "Max Label Width": x_max,
        "Region Width": crop_image.shape[2],
        "Region Size": crop_image.size,
        "img.shape==msk.shape": image_array.shape == mask_array.shape,
        "img.shape": str(image_array.shape),
        "Min_img": float(np.min(tumor_region)),
        "Max_img": float(np.max(tumor_region)),
        "Avg_img": float(np.mean(tumor_region)),
        "Equal Spacing": image_spacing == mask_spacing,
        "spacing[0]": image_spacing[0],
        "spacing[1]": image_spacing[1],
        "spacing[2]": image_spacing[2],
    }

    return row


def save_summary(dataframe, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix.lower() == ".csv":
        dataframe.to_csv(output_file, index=False)
    else:
        dataframe.to_excel(output_file, index=False)


def create_data_summary(args):
    validate_inputs(args.data_root, args.metadata_file)

    metadata = load_metadata(args.metadata_file, args.id_column)
    image_mask_pairs = find_image_mask_pairs(args.data_root, args.image_glob, args.mask_glob)

    logging.info(f"Number of matched image and mask pairs: {len(image_mask_pairs)}")

    rows = []
    problematic_cases = []

    for sample in tqdm(image_mask_pairs, desc="Creating data summary"):
        try:
            metadata_row = get_metadata_row(metadata, sample["id"])
            rows.append(analyze_sample(sample, metadata_row, args))
        except Exception as error:
            problematic_cases.append(
                {
                    "id": sample["id"],
                    "image": str(sample["image"]),
                    "mask": str(sample["mask"]),
                    "error": str(error),
                }
            )

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    save_summary(summary_df, args.output_file)

    if len(problematic_cases) > 0:
        problematic_file = args.output_file.parent / "problematic_data_summary_cases.csv"
        pd.DataFrame(problematic_cases).to_csv(problematic_file, index=False)
        logging.warning(f"Problematic cases were saved to: {problematic_file}")

    logging.info(f"Summary table was saved to: {args.output_file}")


def main():
    args = parse_args()
    configure_logging()
    create_data_summary(args)


if __name__ == "__main__":
    main()