"""Create per-case HDF5 CT targets and deterministic train/test split files."""

import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Directory containing <case>/ct_file.mha.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1225)
    parser.add_argument("--threshold", type=float, default=1224.0)
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite matching HDF5/split files. Directories are never deleted.",
    )
    return parser.parse_args()


def extract_non_noise(volume, threshold):
    return np.where(volume > threshold, volume, 0)


def process_scan(case_directory, output_root, threshold, overwrite):
    case_directory = Path(case_directory)
    case_id = case_directory.name
    output_directory = output_root / case_id
    output_path = output_directory / "ct_xray_data.h5"
    if output_path.exists() and not overwrite:
        return case_id, "skipped"

    try:
        volume_path = case_directory / "ct_file.mha"
        image = sitk.ReadImage(str(volume_path))
        volume = sitk.GetArrayFromImage(image)
        volume = extract_non_noise(volume, threshold)

        output_directory.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as hdf5:
            hdf5.create_dataset("ct", data=volume, compression="gzip")
        return case_id, "created"
    except Exception as error:
        return case_id, f"error: {error}"


def write_split(path, case_ids, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Split file already exists: {path}")
    path.write_text("".join(f"{case_id}\n" for case_id in case_ids), encoding="utf-8")


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if output_root in (input_root, split_root) or input_root == split_root:
        raise ValueError("Input, output, and split directories must be different.")

    output_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose an empty directory or pass --overwrite."
        )

    case_directories = sorted(
        path for path in input_root.iterdir() if (path / "ct_file.mha").is_file()
    )
    if len(case_directories) < 2:
        raise RuntimeError("At least two input cases are required.")

    case_ids = [path.name for path in case_directories]
    train_ids, test_ids = train_test_split(
        case_ids,
        test_size=args.test_size,
        random_state=args.seed,
    )
    write_split(split_root / "train.txt", sorted(train_ids), args.overwrite)
    write_split(split_root / "test.txt", sorted(test_ids), args.overwrite)

    workers = max(1, min(args.workers, len(case_directories)))
    process = partial(
        process_scan,
        output_root=output_root,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap(process, case_directories),
                total=len(case_directories),
                desc="Writing HDF5",
            )
        )

    for case_id, status in results:
        if status.startswith("error"):
            print(f"{case_id}: {status}")
    failures = sum(status.startswith("error") for _, status in results)
    print(f"Completed: {len(results) - failures}/{len(results)} successful")


if __name__ == "__main__":
    main()
