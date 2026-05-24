"""
Description
This script creates 2D and 3D tumor centered image patches from NIfTI CT images and tumor masks. For each image and mask pair, it finds the axial slice with the largest tumor area, crops the tumor region, applies either zero padding outside the mask or tissue padding around the tumor, resizes the crop by padding or cropping to the requested width and height, and saves the output as NIfTI files.

The expected input structure is:

data_root/
    SiteName1/
        img/
            image files ending with .nii.gz
        msk/
            mask files ending with .nii.gz
    SiteName2/
        img/
        msk/

The output structure will be:

output_root/
    2D/
        zero_pad/
            SiteName/
        tissue_pad_32/
            SiteName/
        tissue_pad_64/
            SiteName/
        tissue_pad_128/
            SiteName/
    3D_3_224_224/
    3D_32_224_224/
    3D_64_224_224/
    3D_128_224_224/

How to use
Install the required packages:

pip install SimpleITK numpy pandas tqdm

Run the script:

python slice_tumor_patches.py \
    --data-root /path/to/RawData/AllCohort \
    --output-root /path/to/DeepLearningBased/AllCohort \
    --sites Lung Liver Brain \
    --modes 2d 3d \
    --depths 3 32 64 128 \
    --wh-size 224 \
    --pad-margins 0 32 64 128

If sites are not provided, the script will automatically use all folders inside data_root that contain both img and msk folders.

By default, 3D crops are centered around the slice with the largest tumor area. To reproduce the original notebook depth indexing more closely, use:

python slice_tumor_patches.py \
    --data-root /path/to/RawData/AllCohort \
    --output-root /path/to/DeepLearningBased/AllCohort \
    --depth-mode legacy
"""

import argparse
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Create 2D and 3D tumor image patches from NIfTI images and masks.")

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", choices=["2d", "3d"], default=["2d", "3d"])
    parser.add_argument("--depths", nargs="+", type=int, default=[3, 32, 64, 128])
    parser.add_argument("--wh-size", type=int, default=224)
    parser.add_argument("--pad-margins", nargs="+", type=int, default=[0, 32, 64, 128])

    parser.add_argument("--crop-mode", choices=["start", "center"], default="start")
    parser.add_argument("--depth-mode", choices=["centered", "legacy"], default="centered")
    parser.add_argument("--save-npy", action="store_true")

    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_inputs(data_root):
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")


def remove_nii_suffix(path):
    name = Path(path).name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return Path(name).stem


def get_subject_id(path):
    stem = remove_nii_suffix(path)
    matches = re.findall(r"\d+", stem)

    if len(matches) == 0:
        return stem

    return matches[-1].zfill(4)


def read_nii_image(file_path):
    image = sitk.ReadImage(str(file_path), sitk.sitkFloat64)
    array = sitk.GetArrayFromImage(image)

    return array, image


def write_nii_image(array, reference_image, output_path, start_index_xyz=None, pad_before_xyz=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.GetImageFromArray(array.astype(np.float32), isVector=False)
    image.SetSpacing(reference_image.GetSpacing())
    image.SetDirection(reference_image.GetDirection())

    if start_index_xyz is None:
        image.SetOrigin(reference_image.GetOrigin())
    else:
        if pad_before_xyz is None:
            pad_before_xyz = [0, 0, 0]

        physical_index = [
            float(start_index_xyz[0] - pad_before_xyz[0]),
            float(start_index_xyz[1] - pad_before_xyz[1]),
            float(start_index_xyz[2] - pad_before_xyz[2]),
        ]

        image.SetOrigin(reference_image.TransformContinuousIndexToPhysicalPoint(physical_index))

    sitk.WriteImage(image, str(output_path))


def save_npy_image(array, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)


def get_label_boundary(mask_array):
    z_positions, y_positions, x_positions = np.where(mask_array > 0)

    if len(z_positions) == 0:
        return None

    return {
        "z_min": int(np.min(z_positions)),
        "z_max": int(np.max(z_positions)),
        "y_min": int(np.min(y_positions)),
        "y_max": int(np.max(y_positions)),
        "x_min": int(np.min(x_positions)),
        "x_max": int(np.max(x_positions)),
    }


def get_max_tumor_slice(mask_array):
    boundary = get_label_boundary(mask_array)

    if boundary is None:
        return None

    z_min = boundary["z_min"]
    z_max = boundary["z_max"]
    slice_sums = np.sum(mask_array[z_min : z_max + 1] > 0, axis=(1, 2))

    return int(z_min + np.argmax(slice_sums))


def get_depth_bounds(max_slice_index, depth_size, number_of_slices, depth_mode):
    if depth_mode == "legacy":
        depth_start = max(0, max_slice_index - ((depth_size // 2) - 1))
        depth_end = min(number_of_slices, depth_start + depth_size)
        return depth_start, depth_end

    depth_start = max(0, max_slice_index - (depth_size // 2))
    depth_end = min(number_of_slices, depth_start + depth_size)
    depth_start = max(0, depth_end - depth_size)

    return depth_start, depth_end


def fit_axis_size(array, axis, target_size, fill_value, crop_mode):
    current_size = array.shape[axis]
    pad_before = 0
    crop_offset = 0

    if current_size < target_size:
        pad_before = (target_size - current_size) // 2
        pad_after = target_size - current_size - pad_before

        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (pad_before, pad_after)

        array = np.pad(
            array,
            pad_width=pad_width,
            mode="constant",
            constant_values=fill_value,
        )

    elif current_size > target_size:
        if crop_mode == "center":
            crop_offset = (current_size - target_size) // 2

        slicer = [slice(None)] * array.ndim
        slicer[axis] = slice(crop_offset, crop_offset + target_size)
        array = array[tuple(slicer)]

    return array, pad_before, crop_offset


def fit_width_height(array, wh_size, fill_value, crop_mode):
    array, pad_y_before, crop_y_offset = fit_axis_size(
        array=array,
        axis=1,
        target_size=wh_size,
        fill_value=fill_value,
        crop_mode=crop_mode,
    )

    array, pad_x_before, crop_x_offset = fit_axis_size(
        array=array,
        axis=2,
        target_size=wh_size,
        fill_value=fill_value,
        crop_mode=crop_mode,
    )

    return array, pad_y_before, pad_x_before, crop_y_offset, crop_x_offset


def pad_depth(array, depth_size, fill_value):
    if array.shape[0] >= depth_size:
        return array[:depth_size]

    depth_pad_length = depth_size - array.shape[0]

    return np.pad(
        array,
        pad_width=((0, depth_pad_length), (0, 0), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )


def get_spatial_crop_bounds(mask_shape, boundary, zero_padding, pad_margin):
    if zero_padding:
        y_start = boundary["y_min"]
        y_end = boundary["y_max"] + 1
        x_start = boundary["x_min"]
        x_end = boundary["x_max"] + 1
    else:
        y_start = max(0, boundary["y_min"] - pad_margin)
        y_end = min(mask_shape[1], boundary["y_max"] + 1 + pad_margin)
        x_start = max(0, boundary["x_min"] - pad_margin)
        x_end = min(mask_shape[2], boundary["x_max"] + 1 + pad_margin)

    return y_start, y_end, x_start, x_end


def create_2d_patch(
    image_path,
    mask_path,
    output_dir,
    wh_size,
    zero_padding,
    pad_margin,
    crop_mode,
    save_npy,
):
    image_array, image_header = read_nii_image(image_path)
    mask_array, _ = read_nii_image(mask_path)

    boundary = get_label_boundary(mask_array)

    if boundary is None:
        raise ValueError("Mask is empty")

    max_slice_index = get_max_tumor_slice(mask_array)
    fill_value = float(np.min(image_array))

    y_start, y_end, x_start, x_end = get_spatial_crop_bounds(
        mask_shape=mask_array.shape,
        boundary=boundary,
        zero_padding=zero_padding,
        pad_margin=pad_margin,
    )

    if zero_padding:
        source_array = np.where(mask_array > 0, image_array, fill_value)
    else:
        source_array = image_array

    crop_array = source_array[
        max_slice_index : max_slice_index + 1,
        y_start:y_end,
        x_start:x_end,
    ]

    crop_array, pad_y_before, pad_x_before, crop_y_offset, crop_x_offset = fit_width_height(
        array=crop_array,
        wh_size=wh_size,
        fill_value=fill_value,
        crop_mode=crop_mode,
    )

    subject_id = get_subject_id(image_path)
    padding_name = "zero_pad" if zero_padding else f"tissue_pad_{pad_margin}"
    output_name = f"{subject_id}_2Dcrop_{wh_size}_{wh_size}_{padding_name}.nii.gz"
    output_path = output_dir / output_name

    write_nii_image(
        array=crop_array,
        reference_image=image_header,
        output_path=output_path,
        start_index_xyz=[
            x_start + crop_x_offset,
            y_start + crop_y_offset,
            max_slice_index,
        ],
        pad_before_xyz=[
            pad_x_before,
            pad_y_before,
            0,
        ],
    )

    if save_npy:
        save_npy_image(crop_array, output_path.with_suffix("").with_suffix(".npy"))

    return output_path


def create_3d_patch(
    image_path,
    mask_path,
    output_dir,
    depth_size,
    wh_size,
    zero_padding,
    pad_margin,
    crop_mode,
    depth_mode,
    save_npy,
):
    image_array, image_header = read_nii_image(image_path)
    mask_array, _ = read_nii_image(mask_path)

    boundary = get_label_boundary(mask_array)

    if boundary is None:
        raise ValueError("Mask is empty")

    max_slice_index = get_max_tumor_slice(mask_array)
    depth_start, depth_end = get_depth_bounds(
        max_slice_index=max_slice_index,
        depth_size=depth_size,
        number_of_slices=mask_array.shape[0],
        depth_mode=depth_mode,
    )

    fill_value = float(np.min(image_array))

    y_start, y_end, x_start, x_end = get_spatial_crop_bounds(
        mask_shape=mask_array.shape,
        boundary=boundary,
        zero_padding=zero_padding,
        pad_margin=pad_margin,
    )

    if zero_padding:
        source_array = np.where(mask_array > 0, image_array, fill_value)
    else:
        source_array = image_array

    crop_array = source_array[
        depth_start:depth_end,
        y_start:y_end,
        x_start:x_end,
    ]

    crop_array, pad_y_before, pad_x_before, crop_y_offset, crop_x_offset = fit_width_height(
        array=crop_array,
        wh_size=wh_size,
        fill_value=fill_value,
        crop_mode=crop_mode,
    )

    crop_array = pad_depth(
        array=crop_array,
        depth_size=depth_size,
        fill_value=fill_value,
    )

    subject_id = get_subject_id(image_path)
    padding_name = "zero_pad" if zero_padding else f"tissue_pad_{pad_margin}"
    output_name = f"{subject_id}_3Dcrop_{depth_size}_{wh_size}_{wh_size}_{padding_name}.nii.gz"
    output_path = output_dir / output_name

    write_nii_image(
        array=crop_array,
        reference_image=image_header,
        output_path=output_path,
        start_index_xyz=[
            x_start + crop_x_offset,
            y_start + crop_y_offset,
            depth_start,
        ],
        pad_before_xyz=[
            pad_x_before,
            pad_y_before,
            0,
        ],
    )

    if save_npy:
        save_npy_image(crop_array, output_path.with_suffix("").with_suffix(".npy"))

    return output_path


def infer_sites(data_root):
    if (data_root / "img").is_dir() and (data_root / "msk").is_dir():
        return ["."]

    sites = []

    for folder in sorted(data_root.iterdir()):
        if folder.is_dir() and (folder / "img").is_dir() and (folder / "msk").is_dir():
            sites.append(folder.name)

    return sites


def get_site_directory(data_root, site):
    if site == ".":
        return data_root

    return data_root / site


def get_site_output_directory(output_root, task_folder, padding_name, site):
    if site == ".":
        return output_root / task_folder / padding_name

    return output_root / task_folder / padding_name / site


def find_image_mask_pairs(site_directory):
    image_paths = sorted((site_directory / "img").glob("*.nii.gz"))
    mask_paths = sorted((site_directory / "msk").glob("*.nii.gz"))

    masks_by_id = defaultdict(list)

    for mask_path in mask_paths:
        masks_by_id[get_subject_id(mask_path)].append(mask_path)

    pairs = []
    missing_masks = []

    for image_path in image_paths:
        subject_id = get_subject_id(image_path)
        matched_masks = masks_by_id.get(subject_id, [])

        if len(matched_masks) == 0:
            missing_masks.append(str(image_path))
            continue

        pairs.append(
            {
                "id": subject_id,
                "image": image_path,
                "mask": matched_masks[0],
            }
        )

    if len(pairs) == 0 and len(image_paths) == len(mask_paths):
        pairs = [
            {
                "id": get_subject_id(image_path),
                "image": image_path,
                "mask": mask_path,
            }
            for image_path, mask_path in zip(image_paths, mask_paths)
        ]

    return pairs, missing_masks


def run_slicing(args):
    validate_inputs(args.data_root)

    sites = args.sites if args.sites is not None else infer_sites(args.data_root)

    if len(sites) == 0:
        raise FileNotFoundError(f"No valid site folders were found in: {args.data_root}")

    problematic_cases = []

    for site in sites:
        site_directory = get_site_directory(args.data_root, site)
        image_mask_pairs, missing_masks = find_image_mask_pairs(site_directory)

        if len(missing_masks) > 0:
            logging.warning(f"{site}: {len(missing_masks)} images did not have matched masks")

        logging.info(f"{site}: {len(image_mask_pairs)} image and mask pairs found")

        for pair in tqdm(image_mask_pairs, desc=f"Slicing {site}"):
            for pad_margin in args.pad_margins:
                zero_padding = pad_margin == 0
                padding_name = "zero_pad" if zero_padding else f"tissue_pad_{pad_margin}"

                try:
                    if "2d" in args.modes:
                        output_dir = get_site_output_directory(
                            output_root=args.output_root,
                            task_folder="2D",
                            padding_name=padding_name,
                            site=site,
                        )

                        create_2d_patch(
                            image_path=pair["image"],
                            mask_path=pair["mask"],
                            output_dir=output_dir,
                            wh_size=args.wh_size,
                            zero_padding=zero_padding,
                            pad_margin=pad_margin,
                            crop_mode=args.crop_mode,
                            save_npy=args.save_npy,
                        )

                    if "3d" in args.modes:
                        for depth_size in args.depths:
                            task_folder = f"3D_{depth_size}_{args.wh_size}_{args.wh_size}"
                            output_dir = get_site_output_directory(
                                output_root=args.output_root,
                                task_folder=task_folder,
                                padding_name=padding_name,
                                site=site,
                            )

                            create_3d_patch(
                                image_path=pair["image"],
                                mask_path=pair["mask"],
                                output_dir=output_dir,
                                depth_size=depth_size,
                                wh_size=args.wh_size,
                                zero_padding=zero_padding,
                                pad_margin=pad_margin,
                                crop_mode=args.crop_mode,
                                depth_mode=args.depth_mode,
                                save_npy=args.save_npy,
                            )

                except Exception as error:
                    problematic_cases.append(
                        {
                            "site": site,
                            "id": pair["id"],
                            "image": str(pair["image"]),
                            "mask": str(pair["mask"]),
                            "padding": padding_name,
                            "error": str(error),
                        }
                    )

    if len(problematic_cases) > 0:
        args.output_root.mkdir(parents=True, exist_ok=True)
        problematic_file = args.output_root / "problematic_slicing_cases.csv"
        pd.DataFrame(problematic_cases).to_csv(problematic_file, index=False)
        logging.warning(f"Problematic cases were saved to: {problematic_file}")

    logging.info("Finished")


def main():
    args = parse_args()
    configure_logging()
    run_slicing(args)


if __name__ == "__main__":
    main()