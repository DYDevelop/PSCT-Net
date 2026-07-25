"""Generate AP and lateral DRRs from resized CT volumes."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

# Allow `python dataset_gen/make_drr_320.py` from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.config.config import cfg, merge_dict_and_yaml
from lib.model.nets.generator.drr_projector_new import DRRProjector
from lib.utils.transform_3d import (
    Compose,
    Limit_Min_Max_Threshold,
    Normalization,
    Normalization_gaussian,
    ToTensor,
)


VOLUME_SHAPE = (320, 320, 320)
DETECTOR_SHAPE = (320, 320)
VIEWS = [
    ("AP", (-1.5708, 0.0, 0.0), "xray1"),
    ("LL", (0.0, 0.0, 0.0), "xray2"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Directory containing <case>/ct_file.mha.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device. The differentiable projector normally requires CUDA.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite matching PNG files. Existing directories are never deleted.",
    )
    return parser.parse_args()


def save_tensor_as_image(tensor, filepath):
    tensor = tensor.squeeze().detach().cpu()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    tensor = (tensor * 255).clamp(0, 255).byte()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor.numpy(), mode="L").save(filepath)


def load_scan(path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def build_ct_transform(options):
    return Compose(
        [
            Limit_Min_Max_Threshold(options.CT_MIN_MAX[0], options.CT_MIN_MAX[1]),
            Normalization(options.CT_MIN_MAX[0], options.CT_MIN_MAX[1]),
            Normalization_gaussian(
                options.CT_MEAN_STD[0], options.CT_MEAN_STD[1]
            ),
            ToTensor(),
        ]
    )


def make_drr_for_volume(
    volume_path,
    output_root,
    projector,
    ct_transform,
    device,
    overwrite,
):
    case_id = volume_path.parent.name
    output_paths = [
        output_root / f"{case_id}_{suffix}.png" for _, _, suffix in VIEWS
    ]
    if all(path.exists() for path in output_paths) and not overwrite:
        return False

    ct_scan = load_scan(volume_path).astype(np.float32)
    ct_normal = ct_transform(ct_scan)
    if tuple(ct_normal.shape) != VOLUME_SHAPE:
        raise ValueError(
            f"Expected volume shape {VOLUME_SHAPE}, got {tuple(ct_normal.shape)}"
        )

    ct_normal = ct_normal.unsqueeze(0).unsqueeze(0).to(device)
    for (_, angles, _), output_path in zip(VIEWS, output_paths):
        if output_path.exists() and not overwrite:
            continue
        theta = torch.tensor(
            [angles], device=ct_normal.device, dtype=ct_normal.dtype
        )
        transform = torch.cat([theta, torch.zeros_like(theta)], dim=1)
        projection = projector(ct_normal, transform_param=transform)
        save_tensor_as_image(projection, output_path)
    return True


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if input_root == output_root:
        raise ValueError("Input and output directories must be different.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose an empty directory or pass --overwrite."
        )

    options = merge_dict_and_yaml(dict(), cfg)
    device = torch.device(args.device)
    projector = DRRProjector(
        mode="forward",
        volume_shape=VOLUME_SHAPE,
        detector_shape=DETECTOR_SHAPE,
        voxel_size=(1.0, 1.0, 1.0),
        pixel_size=(1.5, 1.5),
        step_size=0.1,
        source_to_detector_distance=1500.0,
        isocenter_distance=1000.0,
        interp="nearest",
    ).to(device)
    ct_transform = build_ct_transform(options)

    volumes = sorted(input_root.glob("*/ct_file.mha"))
    if not volumes:
        raise RuntimeError(f"No <case>/ct_file.mha volumes found under {input_root}")

    start = time.time()
    success = 0
    for index, volume_path in enumerate(volumes, start=1):
        case_id = volume_path.parent.name
        try:
            changed = make_drr_for_volume(
                volume_path,
                output_root,
                projector,
                ct_transform,
                device,
                args.overwrite,
            )
            state = "generated" if changed else "skipped"
            print(f"[{index}/{len(volumes)}] {case_id}: {state}")
            success += 1
        except Exception as error:
            print(f"[{index}/{len(volumes)}] {case_id}: ERROR: {error}")

    print(
        f"Completed: {success}/{len(volumes)} successful "
        f"({time.time() - start:.2f}s)"
    )


if __name__ == "__main__":
    main()
