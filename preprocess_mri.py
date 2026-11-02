#!/usr/bin/env python3
"""Export, register, prepare, and inspect prostate MRI.

Dependencies: numpy, scipy, SimpleITK, pydicom, antspyx, Pillow, nibabel.
Public case IDs: Case_ followed by 4-8 digits. Source identifiers are never saved.

Commands (append --help for options):
  python preprocess_mri.py export --input selected_dicom --output public_data --adc-units um2/s
  python preprocess_mri.py register --input public_data/nifti --metadata public_data/metadata --output registered
  python preprocess_mri.py prepare --input registered/nifti --metadata registered/metadata --mask-dir masks --output model_inputs
  python preprocess_mri.py qc --npy model_inputs/npy --output qc_outputs --montages

Export expects curated single-frame MR series in Case_XXXX/{T2WI,DWI,ADC,DCE}.
Each case supplies acquired ADC and a valid, complete DCE series with phase times.
Masks are named Case_XXXX_prostate_mask.nii.gz. Preparation writes Case_XXXX.npy
as float32 [6,128,128,32] CXYZ: T2WI, DWI, ADC, DCE arrival/+10s/+100s targets.
Actual DCE times and approximate selections are recorded. ADC units are explicit.
The segmentation container is optional when masks are supplied. Burned-in pixel
identifiers require separate review; these routines do not perform OCR.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset
from scipy import ndimage as ndi

VERSION = "1.2.1"
CASE_PATTERN = re.compile("Case_[0-9]{4,8}\\Z")
MODALITIES = ("T2WI", "DWI", "ADC", "DCE")
CHANNELS = ("T2WI", "DWI", "ADC", "DCE +0s", "DCE +10s", "DCE +100s")
NUMERIC_ACQUISITION = (
    "MagneticFieldStrength",
    "RepetitionTime",
    "EchoTime",
    "EchoTrainLength",
    "FlipAngle",
    "NumberOfAverages",
    "PixelBandwidth",
    "SliceThickness",
    "SpacingBetweenSlices",
    "TemporalResolution",
    "NumberOfTemporalPositions",
    "AcquisitionDuration",
    "ContrastBolusVolume",
    "MatrixSize",
    "FieldOfView_mm",
    "SliceGap",
    "DiffusionBValues",
)
MANUFACTURERS = ("GE", "Philips", "Siemens", "United Imaging", "Other")
TIMING_METHODS = (
    "AcquisitionDateTime_relative",
    "ContentDateTime_relative",
    "AcquisitionTime_relative",
    "ContentTime_relative",
    "TriggerTime",
    "FrameReferenceTime",
    "TemporalPositionIdentifier",
    "AcquisitionNumber",
    "SeriesNumber/source/phase structural order",
)
GROUPING_METHODS = (
    "acquisition_number",
    "temporal_position_identifier",
    "single_phase",
    "repeated_slice_pattern",
    "instance_number_blocks",
    "acquisition_datetime_abs_s_cluster",
    "content_datetime_abs_s_cluster",
    "acquisition_time_sod_cluster",
    "content_time_sod_cluster",
    "trigger_time_ms_cluster",
    "frame_reference_time_ms_cluster",
)


class ReleaseError(Exception):
    def __init__(self, code):
        if not re.fullmatch("[A-Z][A-Z0-9_]+", code):
            code = "PROCESSING_FAILED"
        self.code = code
        super().__init__(code)


class PublicParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "INVALID_ARGUMENTS: use --help for accepted options.\n")


def case_id(value):
    value = str(value)
    if not CASE_PATTERN.fullmatch(value):
        raise ReleaseError("INVALID_PUBLIC_CASE_ID")
    return value


def require(condition, code):
    if not condition:
        raise ReleaseError(code)


def safe_main(main):
    try:
        return int(main() or 0)
    except ReleaseError as exc:
        print(exc.code, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("INTERRUPTED", file=sys.stderr)
        return 130
    except Exception:
        print("PROCESSING_FAILED", file=sys.stderr)
        return 1


@contextlib.contextmanager
def quiet_libraries():
    """Suppress library diagnostics that may contain private source values."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    old_logging = logging.root.manager.disable
    try:
        with open(os.devnull, "w") as sink, warnings.catch_warnings():
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            logging.disable(logging.CRITICAL)
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)
        logging.disable(old_logging)


def separated_roots(input_root, output_root):
    a, b = (Path(input_root).resolve(), Path(output_root).resolve())
    require(a != b and a not in b.parents and (b not in a.parents), "OVERLAPPING_ROOTS")


def discover_cases(root, requested=None):
    root = Path(root)
    require(root.is_dir(), "INPUT_DIRECTORY_MISSING")
    files = list(root.glob("*.nii.gz"))
    ids = set()
    for path in files:
        for modality in MODALITIES:
            suffix = f"_{modality}.nii.gz"
            if path.name.endswith(suffix):
                ids.add(case_id(path.name[: -len(suffix)]))
    require(bool(ids), "NO_PUBLIC_CASES")
    if requested:
        wanted = {case_id(v) for v in requested}
        require(wanted <= ids, "REQUESTED_CASE_MISSING")
        ids = wanted
    return sorted(ids)


def image_paths(root, cid):
    cid = case_id(cid)
    paths = {m: Path(root) / f"{cid}_{m}.nii.gz" for m in MODALITIES}
    require(all(p.is_file() for p in paths.values()), "MODALITY_MISSING")
    return paths


def clean_image(image):
    result = sitk.GetImageFromArray(sitk.GetArrayFromImage(image), isVector=False)
    result.CopyInformation(image)
    return result


def write_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(clean_image(image), str(path), True)


def read_image(path, dimension=None):
    image = sitk.ReadImage(str(path))
    require(image.GetNumberOfComponentsPerPixel() == 1, "VECTOR_IMAGE_UNSUPPORTED")
    if dimension is not None:
        require(image.GetDimension() == dimension, "INVALID_IMAGE_DIMENSION")
    require(
        all(s > 0 and np.isfinite(s) for s in image.GetSpacing()), "INVALID_SPACING"
    )
    require(np.isfinite(sitk.GetArrayViewFromImage(image)).all(), "NONFINITE_IMAGE")
    return clean_image(image)


def same_geometry(a, b):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-05, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=0.0001, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-05, rtol=0)
    )


def phases(image):
    require(
        image.GetDimension() == 4 and image.GetSize()[3] > 0, "INVALID_DCE_DIMENSION"
    )
    direction = np.asarray(image.GetDirection()).reshape(4, 4)
    require(
        np.allclose(direction[:3, 3], 0) and np.allclose(direction[3, :3], 0),
        "SPATIOTEMPORAL_DIRECTION_UNSUPPORTED",
    )
    return [
        sitk.Extract(image, list(image.GetSize()[:3]) + [0], [0, 0, 0, i])
        for i in range(image.GetSize()[3])
    ]


def join_phases(images, times=None):
    require(bool(images), "DCE_EMPTY")
    out = sitk.JoinSeries([sitk.Cast(im, sitk.sitkFloat32) for im in images])
    if times is not None and len(times) > 1 and all(t is not None for t in times):
        dt = np.diff(np.asarray(times, dtype=float))
        if np.all(dt > 0) and np.max(np.abs(dt - np.median(dt))) <= 0.05:
            out.SetSpacing(list(out.GetSpacing()[:3]) + [float(np.median(dt))])
    return out


def numeric_value(value):
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        return value if np.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        values = [numeric_value(v) for v in value]
        if values and all(v is not None for v in values):
            return values
    return None


def public_metadata(data, cid):
    """Copy only approved acquisition fields; never copy arbitrary input text."""
    require(isinstance(data, dict), "INVALID_METADATA")
    require(data.get("CaseID", cid) == cid, "METADATA_CASE_MISMATCH")
    out = {"CaseID": case_id(cid), "schema_version": VERSION, "modalities": {}}
    source = data.get("modalities", {})
    require(isinstance(source, dict), "INVALID_METADATA")
    for modality in MODALITIES:
        block = source.get(modality, {})
        require(isinstance(block, dict), "INVALID_METADATA")
        target = {"nifti_file": f"{cid}_{modality}.nii.gz"}
        acquisition = block.get("dicom_public_acquisition_metadata", {})
        require(isinstance(acquisition, dict), "INVALID_METADATA")
        safe = {}
        for key in NUMERIC_ACQUISITION:
            value = numeric_value(acquisition.get(key))
            if value is not None:
                safe[key] = value
        if acquisition.get("Manufacturer") in MANUFACTURERS:
            safe["Manufacturer"] = acquisition["Manufacturer"]
        target["dicom_public_acquisition_metadata"] = safe
        for key in ("DiffusionBValues", "DynamicPhases", "resampled_phase_indices"):
            value = numeric_value(block.get(key))
            if value is not None:
                target[key] = value
        if modality == "ADC":
            target["source"] = "acquired"
            if block.get("units") in ("um2/s", "mm2/s"):
                target["units"] = block["units"]
        if modality == "DCE":
            if block.get("phase_ordering_method") in TIMING_METHODS:
                target["phase_ordering_method"] = block["phase_ordering_method"]
            methods = block.get("phase_grouping_methods")
            if isinstance(methods, list) and all(
                m in GROUPING_METHODS for m in methods
            ):
                target["phase_grouping_methods"] = methods
            if block.get("geometry_interpolation") in ("none", "linear"):
                target["geometry_interpolation"] = block["geometry_interpolation"]
            if block.get("resampled_phase_indices") == []:
                target["resampled_phase_indices"] = []
            times = block.get("phase_times_seconds")
            if times is not None:
                require(isinstance(times, list), "INVALID_DCE_TIMING")
                require(
                    all(
                        t is None
                        or (
                            isinstance(t, (int, float))
                            and (not isinstance(t, bool))
                            and np.isfinite(t)
                            and (t >= 0)
                        )
                        for t in times
                    ),
                    "INVALID_DCE_TIMING",
                )
                target["phase_times_seconds"] = times
        out["modalities"][modality] = target
    return out


def load_metadata(root, cid):
    path = Path(root) / f"{cid}_metadata.json"
    require(path.is_file(), "METADATA_MISSING")
    with path.open(encoding="utf-8") as handle:
        return public_metadata(json.load(handle), cid)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


@contextlib.contextmanager
def staging(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".work_", dir=output_root) as directory:
        yield Path(directory)


def publish(stage, output_root, overwrite=False):
    files = sorted(p for p in Path(stage).rglob("*") if p.is_file())
    targets = [(p, Path(output_root) / p.relative_to(stage)) for p in files]
    for _, target in targets:
        require(not target.is_symlink(), "OUTPUT_SYMLINK_REJECTED")
        require(not target.exists() or overwrite, "OUTPUT_EXISTS")
        require(
            Path(output_root).resolve() in target.resolve().parents,
            "UNSAFE_OUTPUT_PATH",
        )
    for source, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)


def run_cases(ids, output_root, operation, report_name):
    rows = []
    for cid in ids:
        try:
            with quiet_libraries():
                operation(cid)
            rows.append({"CaseID": cid, "Status": "OK", "Code": "OK"})
        except ReleaseError as exc:
            rows.append({"CaseID": cid, "Status": "FAILED", "Code": exc.code})
        except Exception:
            rows.append(
                {"CaseID": cid, "Status": "FAILED", "Code": "PROCESSING_FAILED"}
            )
        print(f"{cid}: {rows[-1]['Code']}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / report_name
    require(not target.is_symlink(), "OUTPUT_SYMLINK_REJECTED")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("CaseID", "Status", "Code"))
        writer.writeheader()
        writer.writerows(rows)
    return int(any(row["Status"] != "OK" for row in rows))


DCE_MIN_SLICES_PER_PHASE = 5
DCE_Z_ROUND_DECIMALS = 3
DCE_MULTIPHASE_FILECOUNT_RATIO = 1.5
DCE_TIME_CLUSTER_TOLERANCE_SECONDS = 1.0
DCE_MIN_PHASE_SLICE_COVERAGE_FRACTION = 0.7


def read_dicom_header(path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def median_numeric(values: Iterable[Any]) -> float | None:
    nums = []
    for value in values:
        try:
            if value is None:
                continue
            number = float(value)
            if np.isfinite(number):
                nums.append(number)
        except Exception:
            continue
    if not nums:
        return None
    return float(np.median(nums))


def _numeric_values_from_any(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        values = [value]
    else:
        try:
            values = list(value)
        except Exception:
            values = [value]
    out: list[float] = []
    for item in values:
        try:
            number = float(item)
            if np.isfinite(number):
                out.append(float(number))
        except Exception:
            continue
    return out


def extract_standard_diffusion_b_values(ds: Dataset) -> list[float]:
    values: list[float] = []
    try:
        for element in ds.iterall():
            if getattr(element, "keyword", "") == "DiffusionBValue" or int(
                element.tag
            ) == int(pydicom.tag.Tag(24, 36999)):
                values.extend(_numeric_values_from_any(element.value))
    except Exception:
        pass
    return values


def extract_series_description_b_values(ds: Dataset) -> list[float]:
    text_values: list[str] = []
    for keyword in ["SeriesDescription", "ProtocolName", "SequenceName"]:
        raw = getattr(ds, keyword, None)
        if raw is not None and str(raw).strip():
            text_values.append(str(raw).strip())
    if not text_values:
        return []
    values: list[float] = []
    explicit_b_pattern = re.compile(
        "(?i)(?:^|[^A-Za-z0-9])b(?:value)?[_=\\-\\s:]*(\\d+(?:[_\\-\\s]+\\d+)*)"
    )
    dwi_number_pattern = re.compile(
        "(?i)(?:^|[^A-Za-z0-9])(?:dwi|diffusion|diff)[_=\\-\\s:,]*(\\d+(?:\\.\\d+)?(?:[_\\-\\s,;/]+\\d+(?:\\.\\d+)?)*)"
    )
    for text_value in text_values:
        for match in explicit_b_pattern.finditer(text_value):
            numeric_run = match.group(1)
            for token in re.split("[_\\-\\s]+", numeric_run):
                if not token:
                    continue
                try:
                    number = float(token)
                    if np.isfinite(number) and 0.0 <= number <= 10000.0:
                        values.append(float(number))
                except Exception:
                    continue
        for match in dwi_number_pattern.finditer(text_value):
            numeric_run = match.group(1)
            for token in re.split("[_\\-\\s,;/]+", numeric_run):
                if not token:
                    continue
                try:
                    number = float(token)
                    if (
                        np.isfinite(number)
                        and 0.0 <= number <= 10000.0
                        and (number == 0.0 or number >= 50.0)
                    ):
                        values.append(float(number))
                except Exception:
                    continue
    return values


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return float(number) if np.isfinite(number) else None
    except Exception:
        return None


def parse_int(value: Any) -> int | None:
    value = parse_float(value)
    return int(value) if value is not None else None


def dicom_time_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "." in text:
            main, frac = text.split(".", 1)
            frac_digits = re.sub("\\D", "", frac)
            fraction = float(f"0.{frac_digits}") if frac_digits else 0.0
        else:
            main = text
            fraction = 0.0
        main = re.sub("\\D", "", main).zfill(6)
        hh = int(main[0:2])
        mm = int(main[2:4])
        ss = int(main[4:6])
        return float(hh * 3600 + mm * 60 + ss) + fraction
    except Exception:
        return None


def dicom_date_to_ordinal(value: Any) -> int | None:
    if value is None:
        return None
    text = re.sub("\\D", "", str(value).strip())
    if len(text) < 8:
        return None
    try:
        return datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
    except Exception:
        return None


def dicom_datetime_to_absolute_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 8:
        return None
    try:
        date_part = re.sub("\\D", "", text[:8])
        rest = text[8:]
        rest = re.split("[+-]", rest, maxsplit=1)[0]
        day = dicom_date_to_ordinal(date_part)
        tod = dicom_time_to_seconds(rest)
        if day is None or tod is None:
            return None
        return float(day) * 86400.0 + float(tod)
    except Exception:
        return None


def date_time_pair_to_absolute_seconds(
    date_value: Any, time_value: Any, fallback_date_value: Any = None
) -> float | None:
    tod = dicom_time_to_seconds(time_value)
    if tod is None:
        return None
    day = dicom_date_to_ordinal(date_value)
    if day is None:
        day = dicom_date_to_ordinal(fallback_date_value)
    if day is None:
        return None
    return float(day) * 86400.0 + float(tod)


def vector_float(value: Any, min_len: int = 0) -> list[float] | None:
    if value is None:
        return None
    try:
        vals = [float(v) for v in list(value)]
        if len(vals) < min_len:
            return None
        return vals
    except Exception:
        return None


def slice_location_scalar_from_ds(ds: Dataset) -> float | None:
    ipp = vector_float(getattr(ds, "ImagePositionPatient", None), min_len=3)
    iop = vector_float(getattr(ds, "ImageOrientationPatient", None), min_len=6)
    if ipp is not None and iop is not None:
        row = np.asarray(iop[:3], dtype=float)
        col = np.asarray(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        if np.linalg.norm(normal) > 0:
            return float(np.dot(np.asarray(ipp[:3], dtype=float), normal))
    if ipp is not None:
        return float(ipp[2])
    return parse_float(getattr(ds, "SliceLocation", None))


@dataclass
class DCERow:
    path: Path
    series_uid: str
    series_number: int | None
    instance_number: int | None
    acquisition_number: int | None
    temporal_position_identifier: int | None
    trigger_time_ms: float | None
    frame_reference_time_ms: float | None
    temporal_resolution_ms: float | None
    acquisition_datetime_abs_s: float | None
    content_datetime_abs_s: float | None
    acquisition_time_sod: float | None
    content_time_sod: float | None
    slice_location_mm: float | None
    image_position_patient: list[float] | None
    image_orientation_patient: list[float] | None
    rows: int | None
    columns: int | None


def read_dce_row(path: Path) -> DCERow | None:
    ds = read_dicom_header(path)
    if ds is None:
        return None
    study_date = getattr(ds, "StudyDate", None)
    series_date = getattr(ds, "SeriesDate", None)
    acquisition_date = getattr(ds, "AcquisitionDate", None)
    content_date = getattr(ds, "ContentDate", None)
    fallback_date = acquisition_date or content_date or series_date or study_date
    acquisition_datetime_abs = dicom_datetime_to_absolute_seconds(
        getattr(ds, "AcquisitionDateTime", None)
    )
    if acquisition_datetime_abs is None:
        acquisition_datetime_abs = date_time_pair_to_absolute_seconds(
            acquisition_date,
            getattr(ds, "AcquisitionTime", None),
            fallback_date_value=fallback_date,
        )
    content_datetime_abs = date_time_pair_to_absolute_seconds(
        content_date,
        getattr(ds, "ContentTime", None),
        fallback_date_value=fallback_date,
    )
    temporal_position = parse_int(getattr(ds, "TemporalPositionIdentifier", None))
    if temporal_position is None:
        temporal_position = parse_int(getattr(ds, "TemporalPositionIndex", None))
    return DCERow(
        path=path,
        series_uid=str(getattr(ds, "SeriesInstanceUID", "") or ""),
        series_number=parse_int(getattr(ds, "SeriesNumber", None)),
        instance_number=parse_int(getattr(ds, "InstanceNumber", None)),
        acquisition_number=parse_int(getattr(ds, "AcquisitionNumber", None)),
        temporal_position_identifier=temporal_position,
        trigger_time_ms=parse_float(getattr(ds, "TriggerTime", None)),
        frame_reference_time_ms=parse_float(getattr(ds, "FrameReferenceTime", None)),
        temporal_resolution_ms=parse_float(getattr(ds, "TemporalResolution", None)),
        acquisition_datetime_abs_s=acquisition_datetime_abs,
        content_datetime_abs_s=content_datetime_abs,
        acquisition_time_sod=dicom_time_to_seconds(
            getattr(ds, "AcquisitionTime", None)
        ),
        content_time_sod=dicom_time_to_seconds(getattr(ds, "ContentTime", None)),
        slice_location_mm=slice_location_scalar_from_ds(ds),
        image_position_patient=vector_float(
            getattr(ds, "ImagePositionPatient", None), 3
        ),
        image_orientation_patient=vector_float(
            getattr(ds, "ImageOrientationPatient", None), 6
        ),
        rows=parse_int(getattr(ds, "Rows", None)),
        columns=parse_int(getattr(ds, "Columns", None)),
    )


def dce_row_sort_for_volume(row: DCERow) -> tuple[Any, ...]:
    return (
        row.slice_location_mm is None,
        row.slice_location_mm if row.slice_location_mm is not None else 1e18,
        row.instance_number is None,
        row.instance_number if row.instance_number is not None else 10**12,
        row.path.name,
    )


def dce_structural_row_sort(row: DCERow) -> tuple[Any, ...]:
    return (
        row.acquisition_number is None,
        row.acquisition_number if row.acquisition_number is not None else 10**12,
        row.temporal_position_identifier is None,
        row.temporal_position_identifier
        if row.temporal_position_identifier is not None
        else 10**12,
        row.instance_number is None,
        row.instance_number if row.instance_number is not None else 10**12,
        row.path.name,
    )


def dce_unique_slice_count(rows: Sequence[DCERow]) -> int:
    vals = []
    for row in rows:
        if row.slice_location_mm is not None:
            vals.append(round(float(row.slice_location_mm), DCE_Z_ROUND_DECIMALS))
    return len(set(vals))


def dce_group_by_attribute(
    rows: Sequence[DCERow], attribute: str
) -> list[tuple[Any, list[DCERow], str]]:
    groups: dict[Any, list[DCERow]] = defaultdict(list)
    for row in rows:
        value = getattr(row, attribute)
        if value is not None:
            groups[value].append(row)
    out = []
    for value, group_rows in groups.items():
        if len(group_rows) >= DCE_MIN_SLICES_PER_PHASE:
            out.append(
                (value, sorted(group_rows, key=dce_row_sort_for_volume), attribute)
            )
    out.sort(key=lambda x: x[0])
    return out if len(out) >= 2 else []


def dce_group_by_time_clusters(
    rows: Sequence[DCERow], attribute: str
) -> list[tuple[Any, list[DCERow], str]]:
    valid = [r for r in rows if getattr(r, attribute) is not None]
    if len(valid) < 2 * DCE_MIN_SLICES_PER_PHASE:
        return []
    valid = sorted(
        valid,
        key=lambda r: (
            float(getattr(r, attribute)),
            r.instance_number if r.instance_number is not None else 10**12,
        ),
    )
    clusters: list[list[DCERow]] = []
    current: list[DCERow] = []
    start_time: float | None = None
    for row in valid:
        value = float(getattr(row, attribute))
        if start_time is None:
            current = [row]
            start_time = value
            continue
        if abs(value - start_time) <= DCE_TIME_CLUSTER_TOLERANCE_SECONDS * (
            1000 if attribute.endswith("_ms") else 1
        ):
            current.append(row)
        else:
            if len(current) >= DCE_MIN_SLICES_PER_PHASE:
                clusters.append(current)
            current = [row]
            start_time = value
    if len(current) >= DCE_MIN_SLICES_PER_PHASE:
        clusters.append(current)
    if len(clusters) < 2:
        return []
    overall_unique_slices = dce_unique_slice_count(valid)
    if overall_unique_slices >= DCE_MIN_SLICES_PER_PHASE:
        required = max(
            DCE_MIN_SLICES_PER_PHASE,
            int(
                math.ceil(overall_unique_slices * DCE_MIN_PHASE_SLICE_COVERAGE_FRACTION)
            ),
        )
        if any(dce_unique_slice_count(cluster) < required for cluster in clusters):
            return []
    out = []
    for cluster in clusters:
        key = median_numeric(getattr(r, attribute) for r in cluster)
        out.append(
            (key, sorted(cluster, key=dce_row_sort_for_volume), f"{attribute}_cluster")
        )
    return out


def dce_group_by_repeated_slice_pattern(
    rows: Sequence[DCERow],
) -> list[tuple[Any, list[DCERow], str]]:
    valid = [r for r in rows if r.slice_location_mm is not None]
    if len(valid) < 2 * DCE_MIN_SLICES_PER_PHASE:
        return []
    unique_slices = dce_unique_slice_count(valid)
    if unique_slices < DCE_MIN_SLICES_PER_PHASE:
        return []
    if len(valid) < unique_slices * DCE_MULTIPHASE_FILECOUNT_RATIO:
        return []
    time_attrs = [
        "acquisition_datetime_abs_s",
        "content_datetime_abs_s",
        "acquisition_time_sod",
        "content_time_sod",
        "trigger_time_ms",
    ]
    sort_attr = None
    for attr in time_attrs:
        if sum(getattr(r, attr) is not None for r in valid) >= len(valid) * 0.8:
            sort_attr = attr
            break
    if sort_attr is not None:
        ordered = sorted(
            valid,
            key=lambda r: (
                getattr(r, sort_attr) is None,
                getattr(r, sort_attr) if getattr(r, sort_attr) is not None else 1e18,
                r.instance_number if r.instance_number is not None else 10**12,
            ),
        )
    else:
        ordered = sorted(valid, key=dce_structural_row_sort)
    groups: list[list[DCERow]] = []
    current: list[DCERow] = []
    seen = set()
    for row in ordered:
        key = round(float(row.slice_location_mm), DCE_Z_ROUND_DECIMALS)
        if key in seen and len(current) >= DCE_MIN_SLICES_PER_PHASE:
            groups.append(current)
            current = []
            seen = set()
        current.append(row)
        seen.add(key)
    if len(current) >= DCE_MIN_SLICES_PER_PHASE:
        groups.append(current)
    groups = [g for g in groups if len(g) >= DCE_MIN_SLICES_PER_PHASE]
    if len(groups) < 2:
        return []
    return [
        (index, sorted(group, key=dce_row_sort_for_volume), "repeated_slice_pattern")
        for index, group in enumerate(groups, start=1)
    ]


def dce_group_by_instance_blocks(
    rows: Sequence[DCERow],
) -> list[tuple[Any, list[DCERow], str]]:
    valid = [r for r in rows if r.instance_number is not None]
    if len(valid) < 2 * DCE_MIN_SLICES_PER_PHASE:
        return []
    unique_slices = dce_unique_slice_count(valid)
    if unique_slices < DCE_MIN_SLICES_PER_PHASE:
        return []
    if len(valid) < unique_slices * DCE_MULTIPHASE_FILECOUNT_RATIO:
        return []
    ordered = sorted(
        valid,
        key=lambda r: (
            r.instance_number if r.instance_number is not None else 10**12,
            r.path.name,
        ),
    )
    groups = []
    for start in range(0, len(ordered), unique_slices):
        group = ordered[start : start + unique_slices]
        if len(group) >= DCE_MIN_SLICES_PER_PHASE:
            groups.append(group)
    if len(groups) < 2:
        return []
    return [
        (index, sorted(group, key=dce_row_sort_for_volume), "instance_number_blocks")
        for index, group in enumerate(groups, start=1)
    ]


@dataclass
class DCEPhase:
    source_index: int
    source_series_number: int | None
    phase_index_within_source: int
    grouping_method: str
    rows: list[DCERow]
    source_public_metadata: dict[str, Any]


def phase_median(phase: DCEPhase, attribute: str) -> float | None:
    return median_numeric(getattr(r, attribute) for r in phase.rows)


def phase_structural_key(phase: DCEPhase) -> tuple[Any, ...]:
    return (
        phase.source_series_number is None,
        phase.source_series_number
        if phase.source_series_number is not None
        else 10**12,
        phase.source_index,
        phase.phase_index_within_source,
    )


def unwrap_clock_values(values: Sequence[float]) -> list[float]:
    out = []
    day_offset = 0.0
    prev = None
    for raw in values:
        value = float(raw) + day_offset
        if prev is not None and value < prev - 12.0 * 3600.0:
            day_offset += 86400.0
            value = float(raw) + day_offset
        out.append(value)
        prev = value
    return out


def choose_phase_order(
    phases: Sequence[DCEPhase],
) -> tuple[list[DCEPhase], str, dict[int, float | None]]:
    structural = sorted(phases, key=phase_structural_key)
    if not structural:
        return ([], "no_phases", {})
    priorities = [
        ("acquisition_datetime_abs_s", "AcquisitionDateTime_relative", False, 1.0),
        ("content_datetime_abs_s", "ContentDateTime_relative", False, 1.0),
        ("acquisition_time_sod", "AcquisitionTime_relative", True, 1.0),
        ("content_time_sod", "ContentTime_relative", True, 1.0),
        ("trigger_time_ms", "TriggerTime", False, 0.001),
        ("frame_reference_time_ms", "FrameReferenceTime", False, 0.001),
    ]
    for attr, label, is_clock, scale in priorities:
        values = [phase_median(p, attr) for p in structural]
        if not values or any(v is None for v in values):
            continue
        numeric = [float(v) for v in values]
        if len(set(round(v, 6) for v in numeric)) < 2:
            continue
        if is_clock:
            numeric = unwrap_clock_values(numeric)
        phase_value = {id(p): numeric[i] for i, p in enumerate(structural)}
        ordered = sorted(
            structural, key=lambda p: (phase_value[id(p)], phase_structural_key(p))
        )
        first_value = phase_value[id(ordered[0])]
        relative = {id(p): (phase_value[id(p)] - first_value) * scale for p in ordered}
        return (ordered, label, relative)
    for attr, label in [
        ("temporal_position_identifier", "TemporalPositionIdentifier"),
        ("acquisition_number", "AcquisitionNumber"),
    ]:
        values = [phase_median(p, attr) for p in structural]
        if values and all(v is not None for v in values):
            if len(set(round(float(v), 6) for v in values)) >= 2:
                phase_value = {
                    id(p): float(values[i]) for i, p in enumerate(structural)
                }
                ordered = sorted(
                    structural,
                    key=lambda p: (phase_value[id(p)], phase_structural_key(p)),
                )
                return (ordered, label, {id(p): None for p in ordered})
    return (
        structural,
        "SeriesNumber/source/phase structural order",
        {id(p): None for p in structural},
    )


def valid_partition(groups, rows):
    used = [id(row) for _, group, _ in groups for row in group]
    if len(used) != len(rows) or set(used) != {id(row) for row in rows}:
        return False
    return all(
        (
            len(group) >= DCE_MIN_SLICES_PER_PHASE
            and dce_unique_slice_count(group) == len(group)
            for _, group, _ in groups
        )
    )


def reconstruct_phases(rows):
    for field in ("acquisition_number", "temporal_position_identifier"):
        groups = dce_group_by_attribute(rows, field)
        if groups and valid_partition(groups, rows):
            return groups
    for field in (
        "acquisition_datetime_abs_s",
        "content_datetime_abs_s",
        "acquisition_time_sod",
        "content_time_sod",
        "trigger_time_ms",
        "frame_reference_time_ms",
    ):
        groups = dce_group_by_time_clusters(rows, field)
        if groups and valid_partition(groups, rows):
            return groups
    for function in (dce_group_by_repeated_slice_pattern, dce_group_by_instance_blocks):
        groups = function(rows)
        if groups and valid_partition(groups, rows):
            return groups
    groups = [(1, sorted(rows, key=dce_row_sort_for_volume), "single_phase")]
    require(valid_partition(groups, rows), "DCE_GROUPING_AMBIGUOUS")
    return groups


def collect_files(root):
    require(root.is_dir(), "MODALITY_DIRECTORY_MISSING")
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ds = read_dicom_header(path)
        if ds is None or not hasattr(ds, "SOPClassUID"):
            require(path.suffix.lower() != ".dcm", "DICOM_HEADER_UNREADABLE")
            continue
        require(str(getattr(ds, "Modality", "")) == "MR", "NON_MR_INPUT")
        require(
            int(getattr(ds, "NumberOfFrames", 1)) == 1, "ENHANCED_DICOM_UNSUPPORTED"
        )
        require(
            str(getattr(ds, "BurnedInAnnotation", "")).upper() != "YES",
            "BURNED_IN_ANNOTATION_FLAGGED",
        )
        require(
            not any(element.tag.group & 65280 == 24576 for element in ds),
            "DICOM_OVERLAY_FLAGGED",
        )
        require(bool(getattr(ds, "SeriesInstanceUID", "")), "SERIES_UID_MISSING")
        result.append((path, ds))
    require(bool(result), "DICOM_FILES_MISSING")
    return result


def read_series(items):
    positions = [slice_location_scalar_from_ds(ds) for _, ds in items]
    require(all(v is not None for v in positions), "SLICE_POSITION_MISSING")
    require(
        len({round(v, 3) for v in positions}) == len(items), "DUPLICATE_SLICE_POSITION"
    )
    order = np.argsort(positions)
    sorted_positions = np.asarray(positions)[order]
    if len(order) > 2:
        step = np.diff(sorted_positions)
        require(
            np.allclose(step, np.median(step), atol=0.1, rtol=0.02),
            "IRREGULAR_SLICE_SPACING",
        )
    first = items[int(order[0])][1]
    orientation = getattr(first, "ImageOrientationPatient", None)
    require(
        orientation is not None and len(orientation) == 6, "SLICE_ORIENTATION_MISSING"
    )
    normal = np.cross(orientation[:3], orientation[3:])
    require(
        np.isclose(np.linalg.norm(normal), 1, atol=0.0001), "SLICE_ORIENTATION_INVALID"
    )
    origins = np.asarray(
        [getattr(ds, "ImagePositionPatient", []) for _, ds in items], dtype=float
    )
    require(origins.shape == (len(items), 3), "SLICE_POSITION_MISSING")
    in_plane = origins - (origins @ normal)[:, None] * normal
    require(np.allclose(in_plane, in_plane[0], atol=0.1, rtol=0), "INPLANE_SLICE_DRIFT")
    for _, ds in items:
        require(
            int(ds.Rows) == int(first.Rows) and int(ds.Columns) == int(first.Columns),
            "INCONSISTENT_SLICE_GEOMETRY",
        )
        for key in ("ImageOrientationPatient", "PixelSpacing"):
            a, b = (getattr(first, key, None), getattr(ds, key, None))
            require(
                a is not None and b is not None and np.allclose(a, b, atol=1e-05),
                "INCONSISTENT_SLICE_GEOMETRY",
            )
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(items[int(i)][0]) for i in order])
    reader.LoadPrivateTagsOff()
    image = reader.Execute()
    require(image.GetDimension() == 3, "INVALID_IMAGE_DIMENSION")
    require(np.isfinite(sitk.GetArrayViewFromImage(image)).all(), "NONFINITE_IMAGE")
    return image


def safe_acquisition(ds):
    result = {}
    for key in NUMERIC_ACQUISITION:
        raw = getattr(ds, key, None)
        if raw is None:
            continue
        try:
            value = (
                [float(v) for v in raw]
                if not isinstance(raw, (str, bytes)) and hasattr(raw, "__iter__")
                else float(raw)
            )
            value = numeric_value(value)
        except (ValueError, TypeError):
            continue
        if value is not None:
            result[key] = value
    text = str(getattr(ds, "Manufacturer", "")).lower()
    manufacturer = "Other"
    for tokens, canonical in (
        (("ge medical", "general electric"), "GE"),
        (("philips",), "Philips"),
        (("siemens",), "Siemens"),
        (("united imaging", "uih"), "United Imaging"),
    ):
        if any(token in text for token in tokens):
            manufacturer = canonical
            break
    result["Manufacturer"] = manufacturer
    result["MatrixSize"] = [int(ds.Rows), int(ds.Columns)]
    if hasattr(ds, "PixelSpacing"):
        result["FieldOfView_mm"] = [
            float(ds.Columns * ds.PixelSpacing[1]),
            float(ds.Rows * ds.PixelSpacing[0]),
        ]
    return result


def export_static(cid, modality, root, stage, adc_units):
    items = collect_files(root)
    require(
        len({str(ds.SeriesInstanceUID) for _, ds in items}) == 1,
        "STATIC_SERIES_AMBIGUOUS",
    )
    bvalues = []
    if modality == "DWI":
        per_file = [extract_standard_diffusion_b_values(ds) for _, ds in items]
        if all(len(values) == 1 for values in per_file):
            highest = max(values[0] for values in per_file)
            items = [
                item for item, values in zip(items, per_file) if values[0] == highest
            ]
            bvalues = [float(highest)]
        else:
            require(not any(per_file), "DWI_B_VALUES_AMBIGUOUS")
            bvalues = sorted(
                {v for _, ds in items for v in extract_series_description_b_values(ds)}
            )
    image = read_series(items)
    info = {
        "nifti_file": f"{cid}_{modality}.nii.gz",
        "dicom_public_acquisition_metadata": safe_acquisition(items[0][1]),
    }
    if bvalues:
        info["DiffusionBValues"] = bvalues
    if modality == "ADC":
        if adc_units == "mm2/s":
            image = sitk.Cast(image, sitk.sitkFloat32) * 1000000.0
        info.update(source="acquired", units="um2/s")
    write_image(image, stage / "nifti" / info["nifti_file"])
    return info


def export_dce(cid, root, stage):
    items = collect_files(root)
    headers = {path: ds for path, ds in items}
    sources = defaultdict(list)
    for path, ds in items:
        row = read_dce_row(path)
        require(row is not None, "DCE_HEADER_UNREADABLE")
        sources[str(ds.SeriesInstanceUID)].append(row)
    ordered_sources = sorted(
        sources.values(),
        key=lambda rows: (
            rows[0].series_number if rows[0].series_number is not None else 10**12,
            min(
                (
                    r.acquisition_datetime_abs_s
                    for r in rows
                    if r.acquisition_datetime_abs_s is not None
                ),
                default=float("inf"),
            ),
            str(rows[0].series_uid),
        ),
    )
    reconstructed = []
    for index, rows in enumerate(ordered_sources):
        for phase_index, (_, phase_rows, method) in enumerate(reconstruct_phases(rows)):
            reconstructed.append(
                DCEPhase(
                    index, rows[0].series_number, phase_index, method, phase_rows, {}
                )
            )
    ordered, method, offsets = choose_phase_order(reconstructed)
    times = [offsets[id(phase)] for phase in ordered]
    images, resampled = ([], [])
    for i, phase in enumerate(ordered):
        image = read_series([(row.path, headers[row.path]) for row in phase.rows])
        if images and (not same_geometry(images[0], image)):
            image = sitk.Resample(
                image, images[0], sitk.Transform(), sitk.sitkLinear, 0, sitk.sitkFloat32
            )
            resampled.append(i)
        images.append(image)
    info = {
        "nifti_file": f"{cid}_DCE.nii.gz",
        "DynamicPhases": len(images),
        "phase_times_seconds": times,
        "phase_ordering_method": method,
        "phase_grouping_methods": [phase.grouping_method for phase in ordered],
        "resampled_phase_indices": resampled,
        "geometry_interpolation": "linear" if resampled else "none",
        "dicom_public_acquisition_metadata": safe_acquisition(
            headers[ordered[0].rows[0].path]
        ),
    }
    write_image(join_phases(images, times), stage / "nifti" / info["nifti_file"])
    return info


def load_sources(args):
    result = {}
    if args.manifest:
        manifest = Path(args.manifest)
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            require(
                set(reader.fieldnames or []) == {"case_id", "source_dir"},
                "INVALID_MANIFEST_COLUMNS",
            )
            for row in reader:
                cid = case_id(row["case_id"])
                require(cid not in result, "DUPLICATE_PUBLIC_CASE_ID")
                directory = Path(row["source_dir"])
                if not directory.is_absolute():
                    directory = manifest.parent / directory
                result[cid] = directory
    else:
        root = Path(args.input)
        require(root.is_dir(), "INPUT_DIRECTORY_MISSING")
        result = {case_id(p.name): p for p in sorted(root.iterdir()) if p.is_dir()}
    require(bool(result), "NO_PUBLIC_CASES")
    for root in result.values():
        separated_roots(root, args.output)
    return result


def export_main(argv=None):
    parser = PublicParser(
        description="Export selected DICOM series as public NIfTI volumes and numeric metadata."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", help="Selected DICOM root containing public case folders."
    )
    source.add_argument(
        "--manifest", help="Private CSV containing case_id,source_dir; never copied."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--adc-units", choices=("um2/s", "mm2/s"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    with quiet_libraries():
        sources = load_sources(args)

    def process(cid):
        root = sources[cid]
        with staging(args.output) as stage:
            metadata = {"CaseID": cid, "schema_version": VERSION, "modalities": {}}
            for modality in MODALITIES[:3]:
                metadata["modalities"][modality] = export_static(
                    cid, modality, root / modality, stage, args.adc_units
                )
            metadata["modalities"]["DCE"] = export_dce(cid, root / "DCE", stage)
            write_json(metadata, stage / "metadata" / f"{cid}_metadata.json")
            publish(stage, args.output, args.overwrite)

    return run_cases(sorted(sources), args.output, process, "export_report.csv")


def to_ants(image, ants):
    return ants.from_numpy(
        np.transpose(
            sitk.GetArrayFromImage(image).astype(np.float32), (2, 1, 0)
        ).copy(),
        origin=image.GetOrigin(),
        spacing=image.GetSpacing(),
        direction=np.asarray(image.GetDirection()).reshape(3, 3),
    )


def from_ants(image, reference):
    result = sitk.GetImageFromArray(np.transpose(image.numpy(), (2, 1, 0)).copy())
    result.CopyInformation(reference)
    return result


def foreground(image, ants, diffusion=False):
    array = image.numpy()
    positive = array[np.isfinite(array) & (array > 0)]
    require(positive.size >= 50, "FOREGROUND_EMPTY")
    if diffusion:
        mask = array >= np.percentile(positive, 35)
    else:
        try:
            mask = ants.threshold_image(image, "Otsu", 1).numpy() > 0
        except Exception:
            mask = array > np.percentile(array[np.isfinite(array)], 20)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5, 3)))
    mask = ndi.binary_fill_holes(mask)
    labeled, count = ndi.label(mask)
    require(count > 0, "FOREGROUND_EMPTY")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    mask = labeled == np.argmax(sizes)
    return ants.from_numpy(
        mask.astype(np.uint8),
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
    )


def pelvic_mask(fixed_mask, ants):
    array = fixed_mask.numpy() > 0
    coords = np.argwhere(array)
    require(len(coords) >= 100, "FOREGROUND_TOO_SMALL")
    center = np.round(coords.mean(axis=0)).astype(int)
    half = np.maximum(
        1, np.round(np.array(array.shape) * [0.6, 0.6, 0.8] / 2).astype(int)
    )
    low, high = (
        np.maximum(0, center - half),
        np.minimum(array.shape, center + half + 1),
    )
    box = np.zeros_like(array)
    box[tuple((slice(a, b) for a, b in zip(low, high)))] = True
    selected = array & box
    if selected.sum() < 100:
        selected = array
    return ants.from_numpy(
        selected.astype(np.uint8),
        origin=fixed_mask.origin,
        spacing=fixed_mask.spacing,
        direction=fixed_mask.direction,
    )


def warp(moving, fixed, transforms, ants, mask=False):
    interpolation = "nearestNeighbor" if mask else "linear"
    if transforms:
        return ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=transforms,
            interpolator=interpolation,
        )
    return ants.resample_image_to_target(moving, fixed, interp_type=interpolation)


def mask_metrics(fixed_mask, moving_mask):
    a, b = (fixed_mask.numpy() > 0, moving_mask.numpy() > 0)
    n = int(b.sum())
    dice = float(2 * np.logical_and(a, b).sum() / max(1, a.sum() + b.sum()))
    distance = None
    if a.any() and b.any():
        delta = (np.argwhere(a).mean(0) - np.argwhere(b).mean(0)) * fixed_mask.spacing
        distance = float(np.linalg.norm(fixed_mask.direction @ delta))
    return {
        "foreground_dice": dice,
        "foreground_voxels": n,
        "foreground_com_shift_mm": distance,
    }


def acceptable(metrics, modality):
    accepted = (
        metrics["foreground_voxels"] >= 100 and metrics["foreground_dice"] >= 0.05
    )
    if modality == "DCE":
        accepted = (
            accepted
            and metrics["foreground_com_shift_mm"] is not None
            and (metrics["foreground_com_shift_mm"] <= 20)
        )
    return accepted


def estimate(
    fixed, moving, fixed_mask, moving_mask, pelvic, modality, work, args, ants
):
    attempts = [("Rigid", True, 0.2), ("Rigid", False, 0.2), ("Rigid", False, 0.5)]
    if modality in ("ADC", "DWI") and (not args.no_affine_fallback):
        attempts += [("Affine", True, 0.2), ("Affine", False, 0.5)]
    for index, (kind, use_mask, sampling) in enumerate(attempts):
        kwargs = dict(
            fixed=fixed,
            moving=moving,
            type_of_transform=kind,
            outprefix=str(work / f"attempt_{index}_"),
            aff_metric="mattes",
            aff_sampling=32,
            aff_random_sampling_rate=sampling,
            aff_iterations=args.iterations,
            aff_shrink_factors=(8, 4, 2, 1),
            aff_smoothing_sigmas=(3, 2, 1, 0),
            mask_all_stages=True,
            singleprecision=True,
            verbose=False,
        )
        if use_mask:
            kwargs.update(mask=pelvic, moving_mask=moving_mask)
        if modality == "DCE":
            kwargs["initial_transform"] = "Identity"
        try:
            result = ants.registration(**kwargs)
            transforms = result["fwdtransforms"]
            metrics = mask_metrics(
                fixed_mask, warp(moving_mask, fixed, transforms, ants, True)
            )
            if acceptable(metrics, modality):
                return (
                    transforms,
                    {
                        "transform_type": kind,
                        "attempt": index + 1,
                        "fallback": False,
                        **metrics,
                    },
                )
        except Exception:
            continue
    policy = args.dce_fallback if modality == "DCE" else args.fallback
    if policy == "header":
        metrics = mask_metrics(fixed_mask, warp(moving_mask, fixed, [], ants, True))
        require(acceptable(metrics, modality), "HEADER_RESAMPLING_REJECTED")
        return ([], {"transform_type": "identity", "fallback": True, **metrics})
    raise ReleaseError("REGISTRATION_REJECTED")


def save_transforms(paths, stage, cid, label):
    saved = []
    for index, path in enumerate(paths):
        require(Path(path).suffix in (".mat", ".h5"), "TRANSFORM_FORMAT_UNSUPPORTED")
        target = stage / "transforms" / cid / f"{label}_{index:02d}{Path(path).suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        saved.append(target)
    return saved


def register_case(cid, args):
    import ants

    paths = image_paths(args.input, cid)
    metadata = load_metadata(args.metadata, cid)
    originals = {m: read_image(paths[m], 4 if m == "DCE" else 3) for m in MODALITIES}
    images = {m: sitk.DICOMOrient(originals[m], "LPS") for m in MODALITIES[:3]}
    dce_images = [sitk.DICOMOrient(image, "LPS") for image in phases(originals["DCE"])]
    times = metadata["modalities"]["DCE"].get("phase_times_seconds")
    require(
        times is None or len(times) == len(dce_images), "DCE_TIMING_LENGTH_MISMATCH"
    )
    fixed = to_ants(images["T2WI"], ants)
    fixed_mask = foreground(fixed, ants)
    pelvic = pelvic_mask(fixed_mask, ants)
    with (
        staging(args.output) as stage,
        tempfile.TemporaryDirectory(prefix="reg_") as tmp,
    ):
        work = Path(tmp)
        report = {
            "CaseID": cid,
            "version": VERSION,
            "reference": "T2WI",
            "threads": args.threads,
            "iterations": list(args.iterations),
            "modalities": {},
        }
        write_image(images["T2WI"], stage / "nifti" / f"{cid}_T2WI.nii.gz")
        adc_transforms = None
        for modality in ("ADC", "DWI"):
            moving = to_ants(images[modality], ants)
            moving_mask = foreground(moving, ants, diffusion=modality == "DWI")
            reuse = (
                modality == "DWI"
                and (not args.no_reuse_adc_transform)
                and adc_transforms
                and same_geometry(images["ADC"], images["DWI"])
            )
            if reuse:
                transforms = adc_transforms
                metrics = mask_metrics(
                    fixed_mask, warp(moving_mask, fixed, transforms, ants, True)
                )
                reuse = acceptable(metrics, modality)
            if reuse:
                info = {"transform_type": "reused_ADC", "fallback": False, **metrics}
            else:
                transforms, info = estimate(
                    fixed,
                    moving,
                    fixed_mask,
                    moving_mask,
                    pelvic,
                    modality,
                    work,
                    args,
                    ants,
                )
                transforms = save_transforms(transforms, stage, cid, modality)
            transforms = [str(p) for p in transforms]
            if modality == "ADC":
                adc_transforms = transforms
            warped = warp(moving, fixed, transforms, ants)
            write_image(
                from_ants(warped, images["T2WI"]),
                stage / "nifti" / f"{cid}_{modality}.nii.gz",
            )
            info["transforms"] = [str(Path(p).relative_to(stage)) for p in transforms]
            report["modalities"][modality] = info
        representative = len(dce_images) // 2
        moving = to_ants(dce_images[representative], ants)
        moving_mask = foreground(moving, ants)
        shared = True
        try:
            transforms, info = estimate(
                fixed, moving, fixed_mask, moving_mask, pelvic, "DCE", work, args, ants
            )
            transforms = save_transforms(transforms, stage, cid, "DCE")
        except ReleaseError:
            if args.dce_fallback != "independent":
                raise
            shared = False
            transforms, info = ([], {"transform_type": "independent", "fallback": True})
        outputs, phase_reports = ([], [])
        for i, image in enumerate(dce_images):
            moving = to_ants(image, ants)
            moving_mask = foreground(moving, ants)
            if shared:
                phase_transforms = [str(p) for p in transforms]
                phase_info = dict(info)
            else:
                phase_transforms, phase_info = estimate(
                    fixed,
                    moving,
                    fixed_mask,
                    moving_mask,
                    pelvic,
                    "DCE",
                    work,
                    args,
                    ants,
                )
                phase_transforms = [
                    str(p)
                    for p in save_transforms(
                        phase_transforms, stage, cid, f"DCE_{i:04d}"
                    )
                ]
            metrics = mask_metrics(
                fixed_mask, warp(moving_mask, fixed, phase_transforms, ants, True)
            )
            phase_info.update(
                metrics,
                phase_index=i,
                qc_review=not acceptable(metrics, "DCE"),
                transforms=[str(Path(p).relative_to(stage)) for p in phase_transforms],
            )
            phase_reports.append(phase_info)
            outputs.append(
                from_ants(warp(moving, fixed, phase_transforms, ants), images["T2WI"])
            )
        write_image(join_phases(outputs, times), stage / "nifti" / f"{cid}_DCE.nii.gz")
        report["modalities"]["DCE"] = {
            "representative_phase_index": representative,
            "shared_transform": shared,
            "phases": phase_reports,
        }
        metadata["processing"] = {"registration_version": VERSION, "reference": "T2WI"}
        write_json(metadata, stage / "metadata" / f"{cid}_metadata.json")
        write_json(report, stage / "reports" / f"{cid}_registration.json")
        publish(stage, args.output, args.overwrite)


def register_main(argv=None):
    parser = PublicParser(
        description="Register public prostate MRI volumes to the T2WI grid."
    )
    parser.add_argument(
        "--input", required=True, help="Directory containing public NIfTI files."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--iterations", type=int, nargs=4, default=(1000, 500, 250, 100)
    )
    parser.add_argument("--fallback", choices=("error", "header"), default="error")
    parser.add_argument(
        "--dce-fallback", choices=("error", "header", "independent"), default="error"
    )
    parser.add_argument("--no-affine-fallback", action="store_true")
    parser.add_argument("--no-reuse-adc-transform", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    require(
        args.threads > 0 and min(args.iterations) >= 0 and (max(args.iterations) > 0),
        "INVALID_REGISTRATION_SETTINGS",
    )
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    separated_roots(args.input, args.output)
    separated_roots(args.metadata, args.output)
    return run_cases(
        discover_cases(args.input, args.cases),
        args.output,
        lambda cid: register_case(cid, args),
        "registration_report.csv",
    )


def resolve_times(block, count):
    times = np.array(block.get("phase_times_seconds"), dtype=float)
    require(times.shape == (count,), "DCE_TIMING_LENGTH_MISMATCH")
    times -= times[0]
    require(
        np.isfinite(times).all() and (count == 1 or np.all(np.diff(times) > 0)),
        "DCE_TIMING_NOT_INCREASING",
    )
    return times


def detect_arrival(signals, policy="baseline"):
    values = np.asarray(signals, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "DCE_CURVE_INVALID")
    if len(values) < 3:
        require(policy == "baseline", "DCE_ARRIVAL_UNDETERMINED")
        return (len(values) - 1, "approximate_sparse_curve")
    for index in range(2, len(values)):
        baseline = values[:index]
        median = np.median(baseline)
        sigma = 1.4826 * np.median(np.abs(baseline - median))
        threshold = max(abs(median) * 0.05, 3 * sigma, 1e-06)
        window = values[index : index + 2]
        if policy == "strict" and len(window) < 2:
            continue
        if (
            values[index] >= median + threshold
            and values[index] > values[index - 1]
            and np.all(window >= median + 0.5 * threshold)
        ):
            return (index, "adaptive_robust_sustained_threshold")
    require(policy == "baseline", "DCE_ARRIVAL_UNDETERMINED")
    differences = np.diff(values)
    relative = differences / np.maximum(np.abs(values[:-1]), 1e-06)
    index = int(np.argmax(relative))
    baseline = values[: max(2, index + 1)]
    median = np.median(baseline)
    threshold = max(
        abs(median) * 0.05, 3 * 1.4826 * np.median(np.abs(baseline - median)), 1e-06
    )
    if differences[index] >= threshold and relative[index] >= 0.05:
        return (index + 1, "maximum_adjacent_relative_rise_fallback")
    if differences[index] > 0:
        return (index + 1, "approximate_maximum_adjacent_rise")
    return (min(2, len(values) - 1), "approximate_first_postbaseline_phase")


def select_phases(times, arrival, policy="baseline"):
    post = np.asarray(times[arrival:]) - times[arrival]
    n = len(post)
    require(n > 0, "DCE_PHASES_MISSING")
    if policy == "baseline" and n <= 3:
        indices = {1: [0, 0, 0], 2: [0, 1, 1], 3: [0, 1, 2]}[n]
        method = f"sparse_{n}_phase_assignment"
    else:
        indices, clipped = ([0], False)
        for target in (10, 100):
            candidates = np.flatnonzero(post >= target - 1e-06)
            if len(candidates):
                indices.append(int(candidates[0]))
            else:
                require(policy == "baseline", "DCE_TARGET_OUTSIDE_ACQUISITION")
                indices.append(n - 1)
                clipped = True
        method = "last_phase_fallback" if clipped else "first_at_or_after_target"
    return ([int(index + arrival) for index in indices], method)


def expanded_bbox(mask, scale):
    require(np.isfinite(scale) and scale >= 1, "INVALID_CROP_SCALE")
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    require(stats.HasLabel(1), "PROSTATE_MASK_EMPTY")
    box = stats.GetBoundingBox(1)
    original_index, original_size = (np.array(box[:3]), np.array(box[3:]))
    center = original_index + (original_size - 1) / 2
    size = np.maximum(1, np.ceil(original_size * scale).astype(int))
    start = np.floor(center - (size - 1) / 2).astype(int)
    end = start + size
    for axis, maximum in enumerate(mask.GetSize()):
        if start[axis] < 0:
            end[axis] -= start[axis]
            start[axis] = 0
        if end[axis] > maximum:
            start[axis] = max(0, start[axis] - (end[axis] - maximum))
            end[axis] = maximum
    return (start.tolist(), (end - start).tolist())


def resize(image, size):
    old_size, old_spacing = (np.array(image.GetSize()), np.array(image.GetSpacing()))
    spacing = old_spacing.copy()
    for axis in range(3):
        if size[axis] > 1 and old_size[axis] > 1:
            spacing[axis] *= (old_size[axis] - 1) / (size[axis] - 1)
    output = sitk.Resample(
        image,
        list(size),
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        spacing.tolist(),
        image.GetDirection(),
        0,
        sitk.sitkFloat32,
    )
    return np.ascontiguousarray(
        np.transpose(sitk.GetArrayFromImage(output), (2, 1, 0)), dtype=np.float32
    )


def normalized(image, mask, stats=None, adc=False, adc_units="um2/s"):
    values = sitk.GetArrayFromImage(image).astype(np.float32)
    require(np.isfinite(values).all(), "NONFINITE_IMAGE")
    if adc:
        if adc_units == "mm2/s":
            values *= 1000000.0
        values = np.clip(values, 0, 2000) / 2000
        details = {"clip_min": 0, "clip_max": 2000, "input_units": adc_units}
    else:
        if stats is None:
            inside = values[sitk.GetArrayViewFromImage(mask) > 0]
            require(inside.size > 0, "PROSTATE_MASK_EMPTY")
            mean, std = (
                float(np.mean(inside, dtype=np.float64)),
                float(np.std(inside, dtype=np.float64)),
            )
        else:
            mean, std = (stats["mean"], stats["std"])
        require(
            np.isfinite(mean) and np.isfinite(std) and (std > 1e-08),
            "PROSTATE_INTENSITY_VARIANCE_INVALID",
        )
        values = ((values - mean) / std).astype(np.float32)
        details = {"mean": mean, "std": std}
    result = sitk.GetImageFromArray(values)
    result.CopyInformation(image)
    return (result, details)


def get_mask(cid, images, args):
    if args.mask_dir:
        mask = read_image(Path(args.mask_dir) / f"{cid}_prostate_mask.nii.gz", 3)
        provenance = {"method": "provided_mask"}
    else:
        require(
            re.fullmatch(
                "[A-Za-z0-9._:/-]+@sha256:[a-f0-9]{64}", args.segmentation_image or ""
            ),
            "SEGMENTATION_IMAGE_DIGEST_REQUIRED",
        )
        with tempfile.TemporaryDirectory(prefix="seg_") as temporary:
            root = Path(temporary)
            inputs, outputs = (root / "input", root / "output")
            outputs.mkdir()
            names = {
                "T2WI": "transverse-t2-prostate-mri",
                "ADC": "transverse-adc-prostate-mri",
                "DWI": "transverse-hbv-prostate-mri",
            }
            for modality, name in names.items():
                write_image(images[modality], inputs / "images" / name / f"{cid}.mha")
            command = [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--memory=12g",
                "--shm-size=12g",
            ]
            if args.gpus:
                command += ["--gpus", args.gpus]
            command += [
                "-v",
                f"{inputs}:/input:ro",
                "-v",
                f"{outputs}:/output",
                args.segmentation_image,
            ]
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=args.segmentation_timeout,
            )
            require(completed.returncode == 0, "SEGMENTATION_CONTAINER_FAILED")
            mask = read_image(
                outputs
                / "images"
                / "transverse-whole-prostate-mri"
                / "prostate_gland.mha",
                3,
            )
        provenance = {
            "method": "container",
            "image_digest": args.segmentation_image.split("@", 1)[1],
        }
    reference = images["T2WI"]
    mask = sitk.Cast(mask > 0.5, sitk.sitkUInt8)
    if not same_geometry(mask, reference):
        mask = sitk.Resample(
            mask,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    require(int(sitk.GetArrayViewFromImage(mask).sum()) > 0, "PROSTATE_MASK_EMPTY")
    return (mask, provenance)


def prepare_case(cid, args):
    paths = image_paths(args.input, cid)
    metadata = load_metadata(args.metadata, cid)
    images = {m: read_image(paths[m], 3) for m in MODALITIES[:3]}
    dynamic = phases(read_image(paths["DCE"], 4))
    reference = images["T2WI"]
    require(
        all(
            same_geometry(reference, image)
            for image in [images["DWI"], images["ADC"], *dynamic]
        ),
        "INPUTS_NOT_REGISTERED",
    )
    adc_units = args.adc_units or metadata["modalities"]["ADC"].get("units")
    require(adc_units in ("um2/s", "mm2/s"), "ADC_UNITS_REQUIRED")
    segmentation_images = dict(images)
    if adc_units == "mm2/s":
        segmentation_images["ADC"] = (
            sitk.Cast(images["ADC"], sitk.sitkFloat32) * 1000000.0
        )
    mask, mask_provenance = get_mask(cid, segmentation_images, args)
    times = resolve_times(metadata["modalities"]["DCE"], len(dynamic))
    inside = sitk.GetArrayViewFromImage(mask) > 0
    signals = [
        float(np.mean(sitk.GetArrayViewFromImage(im)[inside], dtype=np.float64))
        for im in dynamic
    ]
    arrival, arrival_method = detect_arrival(signals, args.arrival_policy)
    indices, phase_method = select_phases(times, arrival, args.dce_policy)
    start, size = expanded_bbox(mask, args.crop_scale)
    cropped_mask = sitk.RegionOfInterest(mask, size, start)
    chosen = [
        images["T2WI"],
        images["DWI"],
        images["ADC"],
        *[dynamic[i] for i in indices],
    ]
    channels, statistics = ([], [])
    for i, image in enumerate(chosen):
        crop = sitk.RegionOfInterest(image, size, start)
        output, stats = normalized(
            crop,
            cropped_mask,
            stats=statistics[3] if i > 3 else None,
            adc=i == 2,
            adc_units=adc_units,
        )
        channels.append(resize(output, tuple(args.size)))
        statistics.append(stats)
    tensor = np.ascontiguousarray(np.stack(channels), dtype=np.float32)
    require(
        tensor.shape == (6, *args.size) and np.isfinite(tensor).all(),
        "INVALID_OUTPUT_TENSOR",
    )
    actual = [float(times[i] - times[arrival]) for i in indices]
    report = {
        "CaseID": cid,
        "version": VERSION,
        "shape": list(tensor.shape),
        "axis_order": "CXYZ",
        "dtype": "float32",
        "channel_names": list(CHANNELS),
        "segmentation": mask_provenance,
        "crop_scale": args.crop_scale,
        "crop_index_xyz": start,
        "crop_size_xyz": size,
        "adc_source": metadata["modalities"]["ADC"].get("source", "unspecified"),
        "intensity_statistics": statistics,
        "arrival_policy": args.arrival_policy,
        "dce_policy": args.dce_policy,
        "phase_times_seconds": times.tolist(),
        "prostate_mean_dce_curve": signals,
        "arrival_phase_index": arrival,
        "arrival_method": arrival_method,
        "selection_method": phase_method,
        "selected_phase_indices": indices,
        "target_seconds_after_arrival": [0, 10, 100],
        "actual_seconds_after_arrival": actual,
        "duplicate_phase_selected": len(set(indices)) < 3,
    }
    with staging(args.output) as stage:
        (stage / "npy").mkdir()
        np.save(stage / "npy" / f"{cid}.npy", tensor, allow_pickle=False)
        write_image(mask, stage / "masks" / f"{cid}_prostate_mask.nii.gz")
        write_image(cropped_mask, stage / "masks" / f"{cid}_prostate_mask_crop.nii.gz")
        write_json(report, stage / "metadata" / f"{cid}_inputs.json")
        publish(stage, args.output, args.overwrite)


def prepare_main(argv=None):
    parser = PublicParser(
        description="Segment, crop, and normalize registered MRI into six-channel model inputs."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    segmentation = parser.add_mutually_exclusive_group(required=True)
    segmentation.add_argument(
        "--mask-dir", help="Public case masks named Case_XXXX_prostate_mask.nii.gz."
    )
    segmentation.add_argument(
        "--segmentation-image",
        help="Locally installed model image pinned by SHA256 digest.",
    )
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--segmentation-timeout", type=int, default=1800)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--adc-units", choices=("um2/s", "mm2/s"))
    parser.add_argument("--crop-scale", type=float, default=1.75)
    parser.add_argument("--size", type=int, nargs=3, default=(128, 128, 32))
    parser.add_argument(
        "--arrival-policy", choices=("baseline", "strict"), default="baseline"
    )
    parser.add_argument(
        "--dce-policy", choices=("baseline", "strict"), default="baseline"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    require(
        min(args.size) >= 2 and args.segmentation_timeout > 0,
        "INVALID_PREPARATION_SETTINGS",
    )
    separated_roots(args.input, args.output)
    separated_roots(args.metadata, args.output)
    if args.mask_dir:
        separated_roots(args.mask_dir, args.output)
    return run_cases(
        discover_cases(args.input, args.cases),
        args.output,
        lambda cid: prepare_case(cid, args),
        "preparation_report.csv",
    )


def check_nifti_header(path):
    image = nib.load(str(path))
    require(len(image.header.extensions) == 0, "NIFTI_EXTENSION_REVIEW_REQUIRED")
    for key in ("descrip", "aux_file", "intent_name"):
        require(
            not bytes(image.header[key]).strip(b"\x00 "), "NIFTI_TEXT_REVIEW_REQUIRED"
        )


def check_metadata_privacy(value):
    blocked_keys = {
        "patientname",
        "patientid",
        "identity",
        "accessionnumber",
        "patientbirthdate",
        "institutionname",
        "institutionaddress",
        "referringphysicianname",
        "studydate",
        "seriesdate",
        "acquisitiondate",
        "acquisitiondatetime",
        "acquisitiontime",
        "contentdate",
        "contenttime",
        "source_dir",
        "filepath",
        "patientfolder",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("_", "")
            require(
                key.lower() not in blocked_keys
                and normalized not in blocked_keys
                and (not normalized.endswith("uid")),
                "METADATA_IDENTIFIER_REVIEW_REQUIRED",
            )
            check_metadata_privacy(child)
    elif isinstance(value, list):
        for child in value:
            check_metadata_privacy(child)
    elif isinstance(value, str):
        require(
            not re.search("(?:^[A-Za-z]:[\\\\/]|^/|\\\\\\\\)", value),
            "METADATA_PATH_REVIEW_REQUIRED",
        )


def geometry(image):
    return {
        "size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def display_window(array):
    values = np.asarray(array)
    values = values[np.isfinite(values) & (values != 0)]
    if not len(values):
        return (0.0, 1.0)
    low, high = np.percentile(values, [1, 99])
    return (float(low), float(high if high > low else low + 1))


def tile(array, label, window, spacing=(1.0, 1.0)):
    low, high = window
    pixels = (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    source = Image.fromarray(np.flipud(pixels))
    physical_width = source.width * spacing[0]
    physical_height = source.height * spacing[1]
    factor = min(180 / physical_width, 180 / physical_height)
    source = source.resize(
        (
            max(1, round(physical_width * factor)),
            max(1, round(physical_height * factor)),
        )
    )
    canvas = Image.new("L", (190, 215), 0)
    canvas.paste(source, ((190 - source.width) // 2, 25 + (180 - source.height) // 2))
    ImageDraw.Draw(canvas).text((5, 5), label, fill=255)
    return canvas


def save_grid(tiles, columns, path):
    canvas = Image.new("L", (190 * columns, 215 * math.ceil(len(tiles) / columns)), 0)
    for index, item in enumerate(tiles):
        canvas.paste(item, (190 * (index % columns), 215 * (index // columns)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def montage_nifti(cid, modality, image, stage):
    array = sitk.GetArrayFromImage(image)
    window = display_window(array)
    spacing = image.GetSpacing()[:2]
    if modality != "DCE":
        tiles = [
            tile(array[z], f"{modality} z={z}", window, spacing)
            for z in range(array.shape[0])
        ]
        save_grid(
            tiles, min(8, len(tiles)), stage / "montages" / f"{cid}_{modality}.png"
        )
        return
    slices = list(range(0, array.shape[1], 3))
    for start in range(0, array.shape[0], 8):
        tiles = [
            tile(array[t, z], f"phase={t} z={z}", window, spacing)
            for t in range(start, min(start + 8, array.shape[0]))
            for z in slices
        ]
        save_grid(tiles, len(slices), stage / "montages" / f"{cid}_DCE_{start:04d}.png")


def inspect_nifti(cid, args, stage):
    paths = image_paths(args.input, cid)
    metadata_path = Path(args.metadata) / f"{cid}_metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        check_metadata_privacy(json.load(handle))
    metadata = load_metadata(args.metadata, cid)
    images = {}
    report = {"CaseID": cid, "version": VERSION, "mode": "nifti", "modalities": {}}
    for modality, path in paths.items():
        check_nifti_header(path)
        image = read_image(path, 4 if modality == "DCE" else 3)
        images[modality] = image
        report["modalities"][modality] = geometry(image)
        if args.montages and modality in args.modalities:
            montage_nifti(cid, modality, image, stage)
    dynamic = phases(images["DCE"])
    block = metadata["modalities"]["DCE"]
    require(
        block.get("DynamicPhases", len(dynamic)) == len(dynamic),
        "DCE_PHASE_COUNT_MISMATCH",
    )
    aligned = all(
        same_geometry(images["T2WI"], image)
        for image in [images["DWI"], images["ADC"], *dynamic]
    )
    if args.require_registered:
        require(aligned, "INPUTS_NOT_REGISTERED")
    report.update(
        registered_geometry=aligned,
        dce_phase_count=len(dynamic),
    )
    return report


def inspect_tensor(cid, args, stage):
    array = np.load(Path(args.npy) / f"{cid}.npy", allow_pickle=False)
    require(
        array.dtype == np.float32 and array.shape == (6, *args.expected_size),
        "INVALID_TENSOR_INTERFACE",
    )
    require(np.isfinite(array).all(), "NONFINITE_TENSOR")
    require(
        array[2].min() >= -1e-06 and array[2].max() <= 1.000001,
        "ADC_TENSOR_RANGE_INVALID",
    )
    if args.montages:
        windows = [display_window(channel) for channel in array[:3]] + [
            display_window(array[3:])
        ] * 3
        z = array.shape[3] // 2
        tiles = [tile(array[i, :, :, z].T, CHANNELS[i], windows[i]) for i in range(6)]
        save_grid(tiles, 3, stage / "montages" / f"{cid}_tensor.png")
    return {
        "CaseID": cid,
        "version": VERSION,
        "mode": "tensor",
        "shape": list(array.shape),
        "dtype": "float32",
        "channel_min": array.min(axis=(1, 2, 3)).tolist(),
        "channel_max": array.max(axis=(1, 2, 3)).tolist(),
    }


def qc_main(argv=None):
    parser = PublicParser(
        description="Check public NIfTI data or model tensors and optionally create montages."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Public NIfTI directory.")
    inputs.add_argument("--npy", help="Model tensor directory.")
    parser.add_argument(
        "--metadata", help="Acquisition JSON directory; required for NIfTI checks."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--montages", action="store_true")
    parser.add_argument(
        "--modalities", nargs="+", choices=MODALITIES, default=MODALITIES
    )
    parser.add_argument("--expected-size", type=int, nargs=3, default=(128, 128, 32))
    parser.add_argument("--require-registered", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    separated_roots(args.input or args.npy, args.output)
    if args.input:
        require(args.metadata is not None, "METADATA_DIRECTORY_REQUIRED")
        separated_roots(args.metadata, args.output)
        ids = discover_cases(args.input, args.cases)
    else:
        ids = sorted(case_id(path.stem) for path in Path(args.npy).glob("*.npy"))
        require(bool(ids), "NO_PUBLIC_CASES")
        if args.cases:
            selected = {case_id(value) for value in args.cases}
            require(selected <= set(ids), "REQUESTED_CASE_MISSING")
            ids = [cid for cid in ids if cid in selected]

    def process(cid):
        with staging(args.output) as stage:
            report = (
                inspect_nifti(cid, args, stage)
                if args.input
                else inspect_tensor(cid, args, stage)
            )
            write_json(report, stage / "reports" / f"{cid}_qc.json")
            publish(stage, args.output, args.overwrite)

    return run_cases(ids, args.output, process, "qc_report.csv")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "export": export_main,
        "register": register_main,
        "prepare": prepare_main,
        "qc": qc_main,
    }
    parser = PublicParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=tuple(commands))
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
    require(bool(argv) and argv[0] in commands, "INVALID_COMMAND")
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(safe_main(main))
