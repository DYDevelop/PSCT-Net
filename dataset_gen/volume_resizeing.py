# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# Modified for the PSCT-Net preprocessing pipeline.
# ------------------------------------------------------------------------------
"""Resample extracted CT volumes to the PSCT-Net input shape."""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pydicom
import scipy.ndimage as ndimage
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument(
        "--dicom-root",
        required=True,
        type=Path,
        help="Original de-identified DICOM root, used only to recover orientation.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite matching output files. Existing directories are never deleted.",
    )
    return parser.parse_args()


def load_scan(path):
    image = sitk.ReadImage(str(path))
    return (
        sitk.GetArrayFromImage(image),
        image.GetOrigin(),
        image.GetSize(),
        image.GetSpacing(),
    )


def save_scan(volume, origin, spacing, path):
    image = sitk.GetImageFromArray(volume, isVector=False)
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetOrigin(tuple(float(value) for value in origin))
    sitk.WriteImage(image, str(path), True)


def resample(image, spacing, new_spacing=(1, 1, 1)):
    spacing = np.asarray(spacing)[::-1]
    new_spacing = np.asarray(new_spacing)[::-1]
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_resize_factor = new_shape / image.shape
    adjusted_spacing = spacing / real_resize_factor
    return (
        ndimage.zoom(image, real_resize_factor, mode="nearest"),
        adjusted_spacing,
    )


def resize_to_standard(scan, target_shape):
    factors = [target / float(current) for target, current in zip(target_shape, scan.shape)]
    return ndimage.zoom(scan, factors, mode="nearest")


def group_dicom_series(case_directory):
    series = {}
    for filename in sorted(case_directory.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(str(filename), stop_before_pixels=True)
            series.setdefault(str(dataset.SeriesInstanceUID), []).append(filename)
        except Exception:
            continue
    return list(series.values())


def find_reference_dicom(mhd_path, dicom_root):
    stem = mhd_path.stem
    try:
        case_id, series_index_text = stem.rsplit("_", 1)
        series_index = int(series_index_text)
    except ValueError as error:
        raise ValueError(
            f"Expected an extracted filename like CASE_0.mhd, got {mhd_path.name}"
        ) from error

    case_directory = dicom_root / case_id
    series = group_dicom_series(case_directory)
    if series_index >= len(series):
        raise IndexError(f"Series {series_index} not found for case {case_id}")
    files = sorted(series[series_index])
    return pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True)


def process_file(task):
    index, input_path, total, dicom_root, output_root, size, overwrite = task
    input_path = Path(input_path)
    case_id = input_path.stem
    output_directory = output_root / case_id
    output_path = output_directory / "ct_file.mha"

    if output_path.exists() and not overwrite:
        return f"Skipped existing: {case_id}"

    try:
        start = time.time()
        scan, _, old_size, spacing = load_scan(input_path)
        scan, new_spacing = resample(scan, spacing, (1, 1, 1))
        scan = resize_to_standard(scan, (size, size, size))

        reference = find_reference_dicom(input_path, dicom_root)
        orientation = np.abs(
            np.round(reference.ImageOrientationPatient[:3]).astype(np.int32)
        )
        if orientation.tolist() == [0, 1, 0]:
            scan = np.transpose(scan, (2, 1, 0))
        scan = np.transpose(scan, (1, 0, 2))

        output_directory.mkdir(parents=True, exist_ok=True)
        save_scan(scan, (0, 0, 0), new_spacing, output_path)
        _, _, new_size, _ = load_scan(output_path)
        elapsed = time.time() - start
        return (
            f"[{index + 1}/{total}] {case_id}: "
            f"{old_size} -> {new_size} ({elapsed:.2f}s)"
        )
    except Exception as error:
        return f"Error processing {case_id}: {error}"


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    dicom_root = args.dicom_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir() or not dicom_root.is_dir():
        raise FileNotFoundError("Both --input-root and --dicom-root must exist.")
    if output_root in (input_root, dicom_root):
        raise ValueError("The output directory must differ from both input directories.")

    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose an empty directory or pass --overwrite."
        )

    input_files = sorted(input_root.glob("*.mhd"))
    if not input_files:
        raise RuntimeError(f"No .mhd files found under {input_root}")

    workers = max(1, min(args.workers, len(input_files)))
    tasks = [
        (
            index,
            str(path),
            len(input_files),
            dicom_root,
            output_root,
            args.size,
            args.overwrite,
        )
        for index, path in enumerate(input_files)
    ]

    with mp.Pool(processes=workers) as pool:
        results = pool.map(process_file, tasks)
    for result in results:
        print(result)

    failures = sum(result.startswith("Error") for result in results)
    print(f"Completed: {len(results) - failures}/{len(results)} successful")


if __name__ == "__main__":
    main()
