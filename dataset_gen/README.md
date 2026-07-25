# Dataset generation

This directory contains the preprocessing pipeline used to create PSCT-Net
training inputs. It contains source code only. Medical images, DICOM headers,
case lists, and generated datasets must not be committed to the repository.

## Privacy requirement

Run the pipeline only on data that has already been de-identified under the
applicable institutional approval and data-use agreement. Input directory names
are reused as case identifiers and therefore must not contain medical-record
numbers, names, dates of birth, or other identifying information.

The scripts do not export complete DICOM headers. Keep every input and output
directory outside the Git checkout; the local `.gitignore` is an additional
safeguard, not a substitute for de-identification and review.

## Usage

Create the repository Conda environment first, then install the two additional
preprocessing dependencies:

```bash
conda env create -f environment.yaml
conda activate PSCT
python -m pip install -r dataset_gen/requirements.txt
```

From the repository root:

```bash
bash dataset_gen/make_dataset.sh \
  /path/to/deidentified_dicom_cases \
  /path/to/private_work_directory
```

Each input case must be a directory containing `.dcm` files. The default series
filter in `ct_extraction.py` is `Facial 3D  2.0  MPR`; pass
`--series-description ""` when invoking that script directly to disable it.

Existing output directories are never deleted. The pipeline stops when an
output directory is non-empty. To overwrite matching generated files without
deleting the directory, append `--overwrite`:

```bash
bash dataset_gen/make_dataset.sh \
  /path/to/deidentified_dicom_cases \
  /path/to/private_work_directory \
  --overwrite
```

Use `python dataset_gen/<script>.py --help` for individual stages and advanced
options such as worker count, split ratio, threshold, and device.

## Outputs

The work directory receives:

- `extracted_mhd/`: converted DICOM series
- `resized_ct/`: 320-cubed volumes
- `xrays/`: AP and lateral DRRs
- `ct_h5/`: per-case CT targets
- `dataset_list/`: deterministic train/test lists
- `final_h5/`: combined CT and two-view HDF5 files

Do not place the work directory inside the repository.
