"""Combine per-case CT targets and two DRRs into final HDF5 files."""

import argparse
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct-root",
        required=True,
        type=Path,
        help="Directory containing <case>/ct_xray_data.h5.",
    )
    parser.add_argument(
        "--xray-root",
        required=True,
        type=Path,
        help="Directory containing <case>_xray1.png and <case>_xray2.png.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite matching output files. Existing directories are never deleted.",
    )
    return parser.parse_args()


def resize_to_standard(scan, target_shape):
    factors = [target / float(current) for target, current in zip(target_shape, scan.shape)]
    return ndimage.zoom(scan, factors, mode="nearest")


def process_case(case_directory, xray_root, output_root, size, overwrite):
    case_directory = Path(case_directory)
    case_id = case_directory.name
    output_directory = output_root / case_id
    output_path = output_directory / "ct_xray_data.h5"
    if output_path.exists() and not overwrite:
        return f"Skipped existing: {case_id}"

    try:
        with h5py.File(case_directory / "ct_xray_data.h5", "r") as hdf5:
            ct = np.asarray(hdf5["ct"])
        ct = resize_to_standard(ct, (size, size, size))

        xrays = []
        for suffix in ("xray1", "xray2"):
            xray_path = xray_root / f"{case_id}_{suffix}.png"
            xray = cv2.imread(str(xray_path), cv2.IMREAD_GRAYSCALE)
            if xray is None:
                raise FileNotFoundError(f"Could not read {xray_path}")
            if float(np.mean(xray)) < 10:
                raise ValueError(f"Projection is nearly blank: {xray_path}")
            xrays.append(
                cv2.normalize(
                    xray,
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                )
            )

        output_directory.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as hdf5:
            hdf5.create_dataset("ct", data=ct, compression="gzip")
            hdf5.create_dataset("xray1", data=xrays[0], compression="gzip")
            hdf5.create_dataset("xray2", data=xrays[1], compression="gzip")
        return f"Success: {case_id}"
    except Exception as error:
        return f"Error processing {case_id}: {error}"


def main():
    args = parse_args()
    ct_root = args.ct_root.expanduser().resolve()
    xray_root = args.xray_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not ct_root.is_dir() or not xray_root.is_dir():
        raise FileNotFoundError("Both --ct-root and --xray-root must exist.")
    if output_root in (ct_root, xray_root):
        raise ValueError("The output directory must differ from both input directories.")

    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose an empty directory or pass --overwrite."
        )

    case_directories = sorted(
        path for path in ct_root.iterdir() if (path / "ct_xray_data.h5").is_file()
    )
    if not case_directories:
        raise RuntimeError(f"No per-case HDF5 files found under {ct_root}")

    workers = max(1, min(args.workers, len(case_directories)))
    process = partial(
        process_case,
        xray_root=xray_root,
        output_root=output_root,
        size=args.size,
        overwrite=args.overwrite,
    )
    start = time.time()
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process, case_directories),
                total=len(case_directories),
                desc="Combining cases",
            )
        )

    success_count = sum(
        result.startswith(("Success", "Skipped")) for result in results
    )
    for result in results:
        if result.startswith("Error"):
            print(result)
    print(
        f"Completed: {success_count}/{len(results)} successful "
        f"({time.time() - start:.2f}s)"
    )


if __name__ == "__main__":
    main()
