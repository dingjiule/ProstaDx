#!/usr/bin/env python3
"""Fit or apply PSA-only and MRI-plus-PSA logistic models.

Dependencies: numpy, scipy, scikit-learn (Python 3.12 tested).
Clinical CSV: CaseID,Label,PSA; PSA must be numeric, nonnegative, in ng/mL.
Public IDs: Case_ followed by 4-8 digits. Labels: 0 or 1.
Splits CSV: CaseID,Label,Split (train,validation,test).
MRI prediction CSVs: CaseID,Label,Probability; exact split membership is required.

  python train_psa_fusion.py train --clinical clinical.csv --splits runs/mri/split_assignment.csv --validation-predictions runs/mri/fusion/validation_predictions.csv --test-predictions runs/mri/fusion/test_predictions.csv --output runs/psa
  python train_psa_fusion.py predict --model runs/psa/models.json --inputs fusion_inputs.csv --output predictions/psa

Prediction inputs: CaseID,PSA,MRIProbability. Run either command with --help.
The log1p-PSA scaler is fitted on training data. Both logistic models are fitted
on validation data, matching the original workflow; validation scores are fitting
scores. Test data do not set coefficients or thresholds. Censored or missing PSA
must be resolved explicitly before use. Models use JSON coefficients, not pickle.
Source paths and raw PSA are excluded from generated prediction tables.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

VERSION = "1.2.0"
CASE_PATTERN = re.compile("Case_[0-9]{4,8}\\Z")


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


def split_digest(rows):
    text = "\n".join(
        f"{r['CaseID']},{r['Label']},{r['Split']}"
        for r in sorted(rows, key=lambda r: r["CaseID"])
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def vectors(labels, probabilities):
    y, p = (np.asarray(labels), np.asarray(probabilities, dtype=float))
    require(
        y.ndim == p.ndim == 1 and len(y) == len(p) and (len(y) > 0),
        "INVALID_PREDICTION_SHAPE",
    )
    require(np.isin(y, (0, 1)).all(), "INVALID_BINARY_LABEL")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "INVALID_PROBABILITY")
    return (y.astype(int), p)


def metrics(labels, probabilities, threshold=0.5):
    y, p = vectors(labels, probabilities)
    require(math.isfinite(threshold), "INVALID_THRESHOLD")
    predicted = p >= threshold
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
        "n": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
    }


def youden_threshold(labels, probabilities, method="cnn"):
    y, p = vectors(labels, probabilities)
    require(len(np.unique(y)) == 2, "THRESHOLD_REQUIRES_BOTH_CLASSES")
    if method == "psa":
        fpr, tpr, thresholds = roc_curve(y, p)
        finite = np.isfinite(thresholds)
        thresholds, scores = (thresholds[finite], (tpr - fpr)[finite])
        candidates = thresholds[np.isclose(scores, scores.max())]
        return float(min(candidates, key=lambda t: abs(t - 0.5)))
    require(method == "cnn", "INVALID_THRESHOLD_METHOD")
    candidates = np.r_[p.min() - 1e-08, np.unique(p), p.max() + 1e-08]
    scores = []
    for threshold in candidates:
        predicted = p >= threshold
        scores.append(float(predicted[y == 1].mean() - predicted[y == 0].mean()))
    best = max(scores)
    tied = [t for t, score in zip(candidates, scores) if abs(score - best) < 1e-12]
    return float(min(tied, key=lambda t: abs(t - 0.5)))


def evaluate_predictions(validation, test, model, method="cnn", fit_validation=False):
    threshold = youden_threshold(
        [r["Label"] for r in validation], [r["Probability"] for r in validation], method
    )
    result = []
    for split, rows in (("validation", validation), ("test", test)):
        for point, value in (("at_0.5", 0.5), ("validation_youden", threshold)):
            result.append(
                {
                    "Model": model,
                    "Split": split,
                    "OperatingPoint": point,
                    "Role": "fitting"
                    if split == "validation" and fit_validation
                    else "selection"
                    if split == "validation"
                    else "held_out",
                    **metrics(
                        [r["Label"] for r in rows],
                        [r["Probability"] for r in rows],
                        value,
                    ),
                }
            )
    return (result, threshold)


def read_predictions(path, expected):
    rows = read_table(path, ("CaseID", "Label", "Probability"))
    unique_cases(rows)
    clean = {
        case_id(r["CaseID"]): {
            "CaseID": case_id(r["CaseID"]),
            "Label": label(r["Label"]),
            "Probability": number(r["Probability"], "INVALID_PROBABILITY"),
        }
        for r in rows
    }
    require(
        {cid: r["Label"] for cid, r in clean.items()}
        == {r["CaseID"]: r["Label"] for r in expected},
        "PREDICTION_COHORT_MISMATCH",
    )
    ordered = [clean[r["CaseID"]] for r in expected]
    vectors([r["Label"] for r in ordered], [r["Probability"] for r in ordered])
    return ordered


def fresh_output(output, inputs):
    output = Path(output)
    require(not output.is_symlink(), "OUTPUT_SYMLINK")
    require(
        not output.exists() or (output.is_dir() and (not any(output.iterdir()))),
        "OUTPUT_EXISTS",
    )
    target = output.resolve()
    for source in inputs:
        source = Path(source).resolve()
        require(target != source and target not in source.parents, "OVERLAPPING_ROOTS")
        if source.is_dir():
            require(source not in target.parents, "OVERLAPPING_ROOTS")


def fit_scaler(training_psa):
    values = np.asarray(training_psa, dtype=float)
    require(
        values.ndim == 1
        and len(values) > 1
        and np.isfinite(values).all()
        and (values >= 0).all(),
        "INVALID_PSA",
    )
    values = np.log1p(values)
    mean, std = (float(values.mean()), float(values.std(ddof=0)))
    require(
        np.isfinite(mean) and np.isfinite(std) and (std > 0), "INVALID_PSA_VARIANCE"
    )
    return {
        "transform": "log1p_then_zscore",
        "mean": mean,
        "std": std,
        "fit_split": "train",
        "n": len(values),
    }


def transform_psa(values, scaler):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all(), "INVALID_PSA")
    mean = number(scaler["mean"], "INVALID_SCALER")
    std = number(scaler["std"], "INVALID_SCALER")
    require(std > 0 and scaler["transform"] == "log1p_then_zscore", "INVALID_SCALER")
    return (np.log1p(values) - mean) / std


def fit_logistic(features, labels, regularization, max_iter):
    require(
        np.isfinite(regularization) and regularization > 0 and (max_iter > 0),
        "INVALID_LOGISTIC_CONFIG",
    )
    require(set(labels) == {0, 1}, "FIT_REQUIRES_BOTH_CLASSES")
    model = LogisticRegression(C=regularization, solver="lbfgs", max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(features, labels)
        except ConvergenceWarning:
            require(False, "LOGISTIC_NOT_CONVERGED")
    return {
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "classes": [0, 1],
        "iterations": int(model.n_iter_[0]),
    }


def logistic_probability(features, model):
    features = np.asarray(features, dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    intercept = number(model["intercept"], "INVALID_LOGISTIC_MODEL")
    require(
        features.ndim == 2
        and coefficients.ndim == 1
        and (features.shape[1] == len(coefficients))
        and (model["classes"] == [0, 1])
        and np.isfinite(features).all()
        and np.isfinite(coefficients).all(),
        "INVALID_LOGISTIC_MODEL",
    )
    return expit(features @ coefficients + intercept)


def fit_models(
    assignments,
    clinical,
    validation,
    test,
    regularization=1.0,
    max_iter=2000,
):
    by_case = {r["CaseID"]: r for r in clinical}
    require(
        {r["CaseID"]: r["Label"] for r in assignments}
        == {r["CaseID"]: r["Label"] for r in clinical},
        "SPLIT_COHORT_MISMATCH",
    )
    training_psa = [
        by_case[r["CaseID"]]["PSA"] for r in assignments if r["Split"] == "train"
    ]
    scaler = fit_scaler(training_psa)
    groups = {}
    for split, rows in (("validation", validation), ("test", test)):
        psa_z = transform_psa([by_case[r["CaseID"]]["PSA"] for r in rows], scaler)
        mri = np.asarray([r["Probability"] for r in rows])
        groups[split] = {
            "rows": rows,
            "psa_only": psa_z[:, None],
            "mri_plus_psa": np.column_stack((mri, psa_z)),
        }
    labels = [r["Label"] for r in validation]
    models = {
        name: fit_logistic(groups["validation"][name], labels, regularization, max_iter)
        for name in ("psa_only", "mri_plus_psa")
    }
    report, thresholds, outputs = ([], {}, {"validation": [], "test": []})
    for name in ("mri_only", "psa_only", "mri_plus_psa"):
        predictions = {}
        for split, group in groups.items():
            rows = group["rows"]
            probabilities = (
                [r["Probability"] for r in rows]
                if name == "mri_only"
                else logistic_probability(group[name], models[name])
            )
            predictions[split] = [
                {"CaseID": r["CaseID"], "Label": r["Label"], "Probability": float(p)}
                for r, p in zip(rows, probabilities)
            ]
            outputs[split].extend({"Model": name, **r} for r in predictions[split])
        values, thresholds[name] = evaluate_predictions(
            predictions["validation"],
            predictions["test"],
            name,
            method="psa",
            fit_validation=name != "mri_only",
        )
        report.extend(values)
    artifact = {
        "format_version": 1,
        "release_version": VERSION,
        "fit_split": "validation",
        "evaluation_split": "test",
        "scaler": scaler,
        "models": models,
        "feature_order": {
            "psa_only": ["PSA_log1p_z"],
            "mri_plus_psa": ["MRIProbability", "PSA_log1p_z"],
        },
        "validation_youden_thresholds": thresholds,
        "split_sha256": split_digest(assignments),
        "logistic_config": {
            "C": regularization,
            "max_iter": max_iter,
            "solver": "lbfgs",
        },
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    return (artifact, outputs, report)


def run_training(args):
    fresh_output(
        args.output,
        [
            args.clinical,
            args.splits,
            args.validation_predictions,
            args.test_predictions,
        ],
    )
    with quiet_libraries():
        clinical = read_cohort(args.clinical, psa=True)
        assignments = read_split(args.splits, clinical)
        validation = read_predictions(
            args.validation_predictions,
            [r for r in assignments if r["Split"] == "validation"],
        )
        test = read_predictions(
            args.test_predictions, [r for r in assignments if r["Split"] == "test"]
        )
        artifact, outputs, report = fit_models(
            assignments,
            clinical,
            validation,
            test,
            args.regularization,
            args.max_iter,
        )
    with staging(args.output) as stage:
        write_json(artifact, stage / "models.json")
        for split, rows in outputs.items():
            write_table(stage / f"{split}_predictions.csv", rows)
        write_table(stage / "metrics.csv", report)
        publish(stage, args.output)
    print("PSA_FUSION_COMPLETE")


def run_prediction(args):
    fresh_output(args.output, [args.model, args.inputs])
    with quiet_libraries():
        artifact = json.loads(args.model.read_text(encoding="utf-8"))
        require(artifact["format_version"] == 1, "UNSUPPORTED_MODEL_FORMAT")
        require(
            artifact["feature_order"]
            == {
                "psa_only": ["PSA_log1p_z"],
                "mri_plus_psa": ["MRIProbability", "PSA_log1p_z"],
            },
            "INVALID_MODEL_FEATURES",
        )
        rows = read_table(args.inputs, ("CaseID", "PSA", "MRIProbability"))
        unique_cases(rows)
        psa = transform_psa(
            [number(r["PSA"], "INVALID_PSA") for r in rows], artifact["scaler"]
        )
        mri = np.array(
            [number(r["MRIProbability"], "INVALID_PROBABILITY") for r in rows]
        )
        require(((mri >= 0) & (mri <= 1)).all(), "INVALID_PROBABILITY")
        psa_probability = logistic_probability(
            psa[:, None], artifact["models"]["psa_only"]
        )
        combined = logistic_probability(
            np.column_stack((mri, psa)), artifact["models"]["mri_plus_psa"]
        )
        predictions = [
            {
                "CaseID": case_id(r["CaseID"]),
                "MRIProbability": float(m),
                "PSAProbability": float(p),
                "MRIPlusPSAProbability": float(c),
            }
            for r, m, p, c in zip(rows, mri, psa_probability, combined)
        ]
    with staging(args.output) as stage:
        write_table(stage / "predictions.csv", predictions)
        publish(stage, args.output)
    print("PREDICTION_COMPLETE")


def main(argv=None):
    parser = PublicParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=PublicParser
    )
    train = commands.add_parser(
        "train", help="Fit on validation; evaluate once on test."
    )
    train.add_argument(
        "--clinical", type=Path, required=True, help="CSV: CaseID,Label,PSA (ng/mL)."
    )
    train.add_argument("--splits", type=Path, required=True)
    train.add_argument("--validation-predictions", type=Path, required=True)
    train.add_argument("--test-predictions", type=Path, required=True)
    train.add_argument(
        "--regularization", type=float, default=1.0, help="Inverse L2 strength C."
    )
    train.add_argument("--max-iter", type=int, default=2000)
    inference = commands.add_parser("predict", help="Apply saved JSON coefficients.")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument(
        "--inputs", type=Path, required=True, help="CSV: CaseID,PSA,MRIProbability."
    )
    for command in (train, inference):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        run_training(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    raise SystemExit(safe_main(main))
