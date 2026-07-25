#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <deidentified-dicom-root> <work-root> [--overwrite]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DICOM_ROOT="$1"
WORK_ROOT="$2"
OVERWRITE="${3:-}"

if [[ "$OVERWRITE" != "" && "$OVERWRITE" != "--overwrite" ]]; then
  echo "The optional third argument must be --overwrite." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "--overwrite" ]]; then
  EXTRA_ARGS+=("--overwrite")
fi

EXTRACTED_ROOT="$WORK_ROOT/extracted_mhd"
RESIZED_ROOT="$WORK_ROOT/resized_ct"
XRAY_ROOT="$WORK_ROOT/xrays"
CT_H5_ROOT="$WORK_ROOT/ct_h5"
SPLIT_ROOT="$WORK_ROOT/dataset_list"
FINAL_ROOT="$WORK_ROOT/final_h5"

python "$SCRIPT_DIR/ct_extraction.py" \
  --input-root "$DICOM_ROOT" \
  --output-root "$EXTRACTED_ROOT" \
  "${EXTRA_ARGS[@]}"

python "$SCRIPT_DIR/volume_resizeing.py" \
  --input-root "$EXTRACTED_ROOT" \
  --dicom-root "$DICOM_ROOT" \
  --output-root "$RESIZED_ROOT" \
  "${EXTRA_ARGS[@]}"

python "$SCRIPT_DIR/make_drr_320.py" \
  --input-root "$RESIZED_ROOT" \
  --output-root "$XRAY_ROOT" \
  "${EXTRA_ARGS[@]}"

python "$SCRIPT_DIR/make_h5.py" \
  --input-root "$RESIZED_ROOT" \
  --output-root "$CT_H5_ROOT" \
  --split-root "$SPLIT_ROOT" \
  "${EXTRA_ARGS[@]}"

python "$SCRIPT_DIR/combine_h5.py" \
  --ct-root "$CT_H5_ROOT" \
  --xray-root "$XRAY_ROOT" \
  --output-root "$FINAL_ROOT" \
  "${EXTRA_ARGS[@]}"
