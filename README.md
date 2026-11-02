# ProstaDx

MRI preprocessing and baseline models for **ProstaDx: A public multicenter prostate multiparametric MRI dataset for clinically significant prostate cancer diagnosis**.

Includes three-channel bpMRI and six-channel mpMRI CNNs, PSA-only logistic regression, and MRI–PSA fusion.

## Installation

Use Python 3.12. Keep all six scripts together.

```bash
pip install torch numpy scipy scikit-learn SimpleITK pydicom antspyx Pillow nibabel
```

## Inputs

- MRI: `Case_XXXX_{T2WI,DWI,ADC,DCE}.nii.gz`, with corresponding `Case_XXXX_metadata.json` files from the export pipeline.
- Masks: `Case_XXXX_prostate_mask.nii.gz`, aligned with registered T2WI.
- Clinical CSV: only `CaseID,Label,PSA` (PSA optional for CNN training; ng/mL). Labels: `0` = non-csPCa, `1` = csPCa.
- Optional split CSV: `CaseID,Label,Split`; splits are `train`, `validation`, and `test`.

## Usage

Run from the script directory. Place the released NIfTI images in `public_data/nifti/` and matching acquisition JSON files in `public_data/metadata/`.

```bash
# Register and prepare MRI using supplied prostate masks
python preprocess_mri.py register --input public_data/nifti --metadata public_data/metadata --output registered
python preprocess_mri.py prepare --input registered/nifti --metadata registered/metadata --mask-dir masks --output model_inputs
python preprocess_mri.py qc --npy model_inputs/npy --output qc_outputs --montages

# Train both MRI baselines
python cnn_train.py train --npy model_inputs/npy --clinical clinical.csv --splits splits.csv --output runs/mri --seed 42

# Fit PSA-only and MRI–PSA models
python train_psa_fusion.py train --clinical clinical.csv --splits runs/mri/split_assignment.csv --validation-predictions runs/mri/fusion/validation_predictions.csv --test-predictions runs/mri/fusion/test_predictions.csv --output runs/psa
```

Tensors are float32 `(6,128,128,32)`: T2WI, DWI, ADC, and DCE at arrival, +10 s, and +100 s. Review preprocessing reports for approximate or repeated DCE phases.

Omit `--splits` for a stratified 70/10/20 split. Seed 42 is illustrative. PSA scaling uses training cases; logistic models fit validation cases and evaluate test cases. Outputs include models, predictions, and metrics. Use new or empty training output directories. Append `--help` for options and inference commands.

## Reproducibility

Research use only.
