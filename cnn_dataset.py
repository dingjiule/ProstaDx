#!/usr/bin/env python3
"""Public cohort tables, stratified splits, and MRI tensor loading."""

from __future__ import annotations

import csv
import hashlib
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from cnn_config import ReleaseError, require
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

CASE_PATTERN = re.compile("Case_[0-9]{4,8}\\Z")


def case_id(value):
    value = str(value)
    if not CASE_PATTERN.fullmatch(value):
        raise ReleaseError("INVALID_PUBLIC_CASE_ID")
    return value


SPLITS = ("train", "validation", "test")


def read_table(path, required, optional=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), "DUPLICATE_COLUMNS")
        require(set(required) <= set(fields), "REQUIRED_COLUMN_MISSING")
        require(set(fields) <= set(required) | set(optional), "UNEXPECTED_COLUMNS")
        rows = list(reader)
    require(bool(rows), "EMPTY_TABLE")
    require(
        all(None not in row and None not in row.values() for row in rows),
        "INVALID_TABLE",
    )
    return rows


def write_table(path, rows):
    require(bool(rows), "EMPTY_TABLE")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value, code):
    try:
        result = float(value)
    except (ValueError, TypeError):
        require(False, code)
    require(math.isfinite(result), code)
    return result


def label(value):
    require(str(value) in ("0", "1"), "INVALID_BINARY_LABEL")
    return int(value)


def unique_cases(rows):
    ids = [case_id(row["CaseID"]) for row in rows]
    require(len(ids) == len(set(ids)), "DUPLICATE_CASE_ID")


def read_cohort(path, psa=False):
    rows = read_table(path, ("CaseID", "Label"), ("PSA",))
    unique_cases(rows)
    clean = []
    for row in rows:
        item = {"CaseID": case_id(row["CaseID"]), "Label": label(row["Label"])}
        if psa:
            item["PSA"] = number(row.get("PSA"), "INVALID_PSA")
            require(item["PSA"] >= 0, "INVALID_PSA")
        clean.append(item)
    return clean


def read_cases(path):
    rows = read_table(path, ("CaseID",))
    unique_cases(rows)
    return [{"CaseID": case_id(row["CaseID"])} for row in rows]


def validate_split(rows):
    unique_cases(rows)
    require(all(row["Split"] in SPLITS for row in rows), "INVALID_SPLIT")
    for split in SPLITS:
        require(
            {row["Label"] for row in rows if row["Split"] == split} == {0, 1},
            "SPLIT_REQUIRES_BOTH_CLASSES",
        )


def read_split(path, cohort=None):
    rows = read_table(path, ("CaseID", "Label", "Split"))
    rows = [
        {
            "CaseID": case_id(r["CaseID"]),
            "Label": label(r["Label"]),
            "Split": r["Split"],
        }
        for r in rows
    ]
    validate_split(rows)
    if cohort is not None:
        expected = {r["CaseID"]: r["Label"] for r in cohort}
        require(
            {r["CaseID"]: r["Label"] for r in rows} == expected, "SPLIT_COHORT_MISMATCH"
        )
    return rows


def make_split(cohort, seed=None):
    unique_cases(cohort)
    n_test, n_val = (round(len(cohort) * 0.2), round(len(cohort) * 0.1))
    require(
        n_test >= 2 and n_val >= 2 and (len(cohort) - n_test - n_val >= 2),
        "COHORT_TOO_SMALL",
    )
    try:
        development, test = train_test_split(
            cohort,
            test_size=n_test,
            random_state=seed,
            stratify=[r["Label"] for r in cohort],
        )
        train, validation = train_test_split(
            development,
            test_size=n_val,
            random_state=seed,
            stratify=[r["Label"] for r in development],
        )
    except ValueError:
        require(False, "STRATIFIED_SPLIT_FAILED")
    rows = [
        {"CaseID": r["CaseID"], "Label": r["Label"], "Split": split}
        for split, group in zip(SPLITS, (train, validation, test))
        for r in group
    ]
    validate_split(rows)
    return rows


def split_digest(rows):
    text = "\n".join(
        f"{r['CaseID']},{r['Label']},{r['Split']}"
        for r in sorted(rows, key=lambda r: r["CaseID"])
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def load_tensor(root, cid, size):
    path = Path(root) / f"{case_id(cid)}.npy"
    require(path.is_file(), "TENSOR_MISSING")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(array.shape == (6, *size), "TENSOR_SHAPE_MISMATCH")
    require(array.dtype == np.float32, "TENSOR_DTYPE_MISMATCH")
    require(np.isfinite(array).all(), "NONFINITE_TENSOR")
    require(
        np.min(array[2]) >= -1e-06 and np.max(array[2]) <= 1 + 1e-06, "ADC_OUT_OF_RANGE"
    )
    return array


class MRIDataset(Dataset):
    def __init__(self, rows, root, stage, size, flip_probability=0):
        require(stage in ("static", "fusion"), "INVALID_MODEL")
        self.rows, self.root, self.stage = (rows, Path(root), stage)
        self.size, self.flip_probability = (tuple(size), flip_probability)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        array = load_tensor(self.root, row["CaseID"], self.size)
        channels = 3 if self.stage == "static" else 6
        tensor = torch.from_numpy(np.array(array[:channels], copy=True, order="C"))
        for axis in (1, 2, 3):
            if self.flip_probability and random.random() < self.flip_probability:
                tensor = tensor.flip(axis)
        return (tensor, int(row.get("Label", -1)), row["CaseID"])


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(rows, root, stage, config, device, seed=None, training=False):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    dataset = MRIDataset(
        rows, root, stage, config.size, config.flip_probability if training else 0
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if seed is not None else None,
        generator=generator,
        persistent_workers=config.workers > 0,
    )
