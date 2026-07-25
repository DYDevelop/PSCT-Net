"""Convert de-identified DICOM series into MetaImage volumes.

This script never copies full DICOM headers to the output directory. Input
directory names are reused as case identifiers, so the input dataset must be
de-identified before this script is run.
"""

import argparse
import os
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Directory containing one de-identified DICOM directory per case.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory in which converted .mhd/.raw files will be written.",
    )
    parser.add_argument(
        "--series-description",
        default="Facial 3D  2.0  MPR",
        help="Only convert series with this exact SeriesDescription. Use an empty value to disable filtering.",
    )
    parser.add_argument("--min-slices", type=int, default=31)
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 8))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite matching output files. Existing directories are never deleted.",
    )
    return parser.parse_args()


def prepare_output_directory(path, overwrite):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Choose an empty directory or pass --overwrite."
        )


def group_dicom_series(case_directory):
    series = {}
    for filename in sorted(glob(os.path.join(case_directory, "*.dcm"))):
        try:
            dataset = pydicom.dcmread(filename, stop_before_pixels=True)
            series.setdefault(str(dataset.SeriesInstanceUID), []).append(filename)
        except Exception:
            continue
    return series


def process_case(task):
    case_directory, output_root, series_description, min_slices, overwrite = task
    case_id = Path(case_directory).name
    results = []

    for series_index, files in enumerate(group_dicom_series(case_directory).values()):
        output_name = f"{case_id}_{series_index}.mhd"
        output_path = output_root / output_name

        if output_path.exists() and not overwrite:
            results.append(f"Skipped existing: {output_name}")
            continue
        if len(files) < min_slices:
            continue

        try:
            # Remove edge slices, matching the original preprocessing protocol.
            dicom_files = sorted(files)[2:-2]
            reference = pydicom.dcmread(dicom_files[len(dicom_files) // 2])
            description = str(getattr(reference, "SeriesDescription", ""))
            if series_description and description != series_description:
                continue

            spacing = (
                float(reference.PixelSpacing[0]),
                float(reference.PixelSpacing[1]),
                float(reference.SliceThickness),
            )
            origin = tuple(float(value) for value in reference.ImagePositionPatient)
            rows, columns = int(reference.Rows), int(reference.Columns)

            slices = []
            for filename in dicom_files:
                dataset = pydicom.dcmread(filename)
                instance_number = int(dataset.InstanceNumber)
                slope = float(getattr(dataset, "RescaleSlope", 1.0))
                intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
                pixels = dataset.pixel_array.astype(np.float32) * slope + intercept
                pixels = cv2.resize(pixels, (columns, rows))
                slices.append((instance_number, pixels + 1024.0))

            slices.sort(key=lambda item: item[0])
            volume = np.stack([pixels for _, pixels in slices], axis=0)
            image = sitk.GetImageFromArray(volume, isVector=False)
            image.SetSpacing(spacing)
            image.SetOrigin(origin)
            sitk.WriteImage(image, str(output_path))
            results.append(f"Converted: {output_name}")
        except Exception as error:
            results.append(f"Error in {case_id}, series {series_index}: {error}")

    return results


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if input_root == output_root:
        raise ValueError("Input and output directories must be different.")

    prepare_output_directory(output_root, args.overwrite)
    case_directories = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not case_directories:
        raise RuntimeError(f"No case directories found under {input_root}")

    workers = max(1, min(args.workers, len(case_directories)))
    tasks = [
        (
            str(case_directory),
            output_root,
            args.series_description,
            args.min_slices,
            args.overwrite,
        )
        for case_directory in case_directories
    ]

    print(f"Processing {len(tasks)} cases with {workers} workers")
    with Pool(processes=workers) as pool:
        for case_results in tqdm(
            pool.imap(process_case, tasks),
            total=len(tasks),
            desc="Converting DICOM",
        ):
            for result in case_results:
                print(result)


if __name__ == "__main__":
    main()
