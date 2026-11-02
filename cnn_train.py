#!/usr/bin/env python3
"""Train or apply three-channel and six-channel residual MRI CNNs.

Dependencies: numpy, scikit-learn, torch (Python 3.12 tested).
Clinical CSV: CaseID,Label with optional PSA; labels are 0 or 1.
Public IDs: Case_ followed by 4-8 digits. Tensors: Case_XXXX.npy, float32 CXYZ
[6,128,128,32], channels T2WI/DWI/ADC/DCE arrival/+10s/+100s; ADC is in [0,1].
Optional splits CSV: CaseID,Label,Split (train,validation,test).

  python cnn_train.py train --npy model_inputs/npy --clinical clinical.csv --splits splits.csv --output runs/mri
  python cnn_train.py predict --checkpoint runs/mri/fusion/best_model.pt --cases cases.csv --npy model_inputs/npy --output predictions/mri

Prediction cases CSV contains CaseID only. Run either command with --help.
Without --splits, generate a random stratified 70/10/20 holdout.
An exact study reproduction needs the original public assignment.
The best validation-AUC checkpoint is evaluated on test; thresholds use validation.
No seed is fixed by default; --seed optionally controls splitting and training.
The static model uses the first three channels; fusion uses
all six in one CNN. Outputs omit source paths and private clinical columns."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sklearn
import torch
from cnn_config import CHANNELS, VERSION, ReleaseError, TrainConfig, require
from cnn_dataset import (
    case_id,
    load_tensor,
    loader,
    make_split,
    read_cases,
    read_cohort,
    read_split,
    split_digest,
    write_table,
)
from cnn_model import SingleBranch3DCNN, model_parameters
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn


class PublicParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "INVALID_ARGUMENTS: use --help for accepted options.\n")


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


def select_device(value):
    require(value in ("auto", "cpu", "cuda"), "INVALID_DEVICE")
    value = (
        "cuda"
        if value == "auto" and torch.cuda.is_available()
        else "cpu"
        if value == "auto"
        else value
    )
    require(value != "cuda" or torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    return torch.device(value)


def configure_runtime(config):
    config.validate()
    torch.set_num_threads(config.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.seed is not None


def seed_all(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, batches, optimizer, criterion, scaler, device, amp):
    model.train()
    total, count = (0.0, 0)
    for images, labels, _ in batches:
        images, labels = (images.to(device), labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(images)["logits"], labels)
        require(bool(torch.isfinite(loss)), "NONFINITE_LOSS")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not amp:
            require(
                all(
                    p.grad is None or bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ),
                "NONFINITE_GRADIENT",
            )
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / count


@torch.inference_mode()
def predict(model, batches, device, criterion=None):
    model.eval()
    rows, loss_sum, count = ([], 0.0, 0)
    for images, labels, ids in batches:
        logits = model(images.to(device))["logits"]
        require(bool(torch.isfinite(logits).all()), "NONFINITE_PREDICTION")
        probabilities = logits.softmax(1)[:, 1].cpu().tolist()
        if criterion is not None:
            loss_sum += float(criterion(logits, labels.to(device))) * len(labels)
            count += len(labels)
        for cid, value, probability in zip(ids, labels.tolist(), probabilities):
            item = {"CaseID": case_id(cid)}
            if value != -1:
                item["Label"] = value
            item["Probability"] = float(probability)
            rows.append(item)
    return (rows, loss_sum / count if count else None)


def train_model(stage, groups, root, output, config, device, assignment_hash):
    seed = config.seed
    seed_all(seed)
    model = SingleBranch3DCNN(**model_parameters(stage, config)).to(device)
    train = loader(groups["train"], root, stage, config, device, seed, training=True)
    validation = loader(
        groups["validation"],
        root,
        stage,
        config,
        device,
        None if seed is None else seed + 1,
    )
    counts = np.bincount([r["Label"] for r in groups["train"]], minlength=2)
    weights = torch.tensor(
        len(groups["train"]) / (2 * counts), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )
    amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_auc, best_epoch, best_state, best_val, stale = (-1.0, 0, None, None, 0)
    log = []
    output.mkdir(parents=True)
    for epoch in range(1, config.epochs + 1):
        with quiet_libraries():
            lr = optimizer.param_groups[0]["lr"]
            loss = train_epoch(model, train, optimizer, criterion, scaler, device, amp)
            values, val_loss = predict(model, validation, device, criterion)
            auc = metrics(
                [r["Label"] for r in values], [r["Probability"] for r in values]
            )["auc"]
            scheduler.step()
        log.append(
            {
                "Epoch": epoch,
                "LearningRate": lr,
                "TrainLoss": loss,
                "ValidationLoss": val_loss,
                "ValidationAUC": auc,
            }
        )
        write_table(output / "epoch_log.csv", log)
        print(f"{stage}: epoch {epoch}, validation AUC {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_epoch, best_val, stale = (auc, epoch, values, 0)
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= config.patience:
            break
    require(best_state is not None, "NO_VALID_CHECKPOINT")
    with quiet_libraries():
        model.load_state_dict(best_state)
        test, _ = predict(
            model,
            loader(
                groups["test"],
                root,
                stage,
                config,
                device,
                None if seed is None else seed + 2,
            ),
            device,
        )
    report, threshold = evaluate_predictions(best_val, test, stage)
    write_table(output / "validation_predictions.csv", best_val)
    write_table(output / "test_predictions.csv", test)
    write_table(output / "metrics.csv", report)
    torch.save(
        {
            "format_version": 1,
            "release_version": VERSION,
            "stage": stage,
            "model_parameters": model_parameters(stage, config),
            "size": list(config.size),
            "channel_order": list(CHANNELS[:3] if stage == "static" else CHANNELS),
            "best_epoch": best_epoch,
            "validation_auc": best_auc,
            "validation_youden_threshold": threshold,
            "split_sha256": assignment_hash,
            "training_config": asdict(config),
            "model_state_dict": best_state,
        },
        output / "best_model.pt",
    )
    write_json(
        {
            "model": stage,
            "best_epoch": best_epoch,
            "validation_auc": best_auc,
            "validation_youden_threshold": threshold,
            "seed": seed,
            "epochs_run": len(log),
        },
        output / "summary.json",
    )
    del train, validation, model
    return report


def run_training(args):
    config = TrainConfig(
        **{field: getattr(args, field) for field in TrainConfig.__dataclass_fields__}
    )
    config = TrainConfig(**{**asdict(config), "size": tuple(config.size)})
    configure_runtime(config)
    sources = [args.npy, args.clinical] + ([args.splits] if args.splits else [])
    fresh_output(args.output, sources)
    with quiet_libraries():
        cohort = read_cohort(args.clinical)
        assignments = (
            read_split(args.splits, cohort)
            if args.splits
            else make_split(cohort, config.seed)
        )
        for row in cohort:
            load_tensor(args.npy, row["CaseID"], config.size)
    groups = {
        split: [r for r in assignments if r["Split"] == split]
        for split in ("train", "validation", "test")
    }
    device = select_device(args.device)
    with staging(args.output) as stage:
        write_table(stage / "split_assignment.csv", assignments)
        report = []
        for name in args.models:
            report.extend(
                train_model(
                    name,
                    groups,
                    args.npy,
                    stage / name,
                    config,
                    device,
                    split_digest(assignments),
                )
            )
        write_table(stage / "metrics.csv", report)
        write_json(
            {
                "release_version": VERSION,
                "config": asdict(config),
                "models": args.models,
                "device": device.type,
                "split_source": "provided" if args.splits else "generated",
                "split_sha256": split_digest(assignments),
                "counts": {split: len(rows) for split, rows in groups.items()},
                "versions": {
                    "torch": str(torch.__version__),
                    "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__,
                },
            },
            stage / "run.json",
        )
        publish(stage, args.output)
    print("TRAINING_COMPLETE")


def run_prediction(args):
    fresh_output(args.output, [args.npy, args.cases, args.checkpoint])
    with quiet_libraries():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        require(checkpoint["format_version"] == 1, "UNSUPPORTED_CHECKPOINT")
        config = TrainConfig(
            **{
                key: value
                for key, value in checkpoint["training_config"].items()
                if key in TrainConfig.__dataclass_fields__
            }
        )
        config = TrainConfig(
            **{
                **asdict(config),
                "size": tuple(checkpoint["size"]),
                "workers": args.workers,
                "batch_size": args.batch_size,
                "threads": args.threads,
            }
        )
        configure_runtime(config)
        name = checkpoint["stage"]
        expected = model_parameters(name, config)
        require(checkpoint["model_parameters"] == expected, "INVALID_CHECKPOINT_CONFIG")
        rows = read_cases(args.cases)
        device = select_device(args.device)
        model = SingleBranch3DCNN(**expected).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        predictions, _ = predict(
            model, loader(rows, args.npy, name, config, device), device
        )
    with staging(args.output) as stage:
        write_table(stage / "predictions.csv", predictions)
        publish(stage, args.output)
    print("PREDICTION_COMPLETE")


def parser():
    result = PublicParser(description=__doc__)
    commands = result.add_subparsers(
        dest="command", required=True, parser_class=PublicParser
    )
    train = commands.add_parser("train", help="Train with one shared held-out split.")
    train.add_argument(
        "--clinical",
        type=Path,
        required=True,
        help="Public CSV: CaseID,Label; optional PSA.",
    )
    train.add_argument("--splits", type=Path, help="Public CSV: CaseID,Label,Split.")
    train.add_argument(
        "--models",
        nargs="+",
        choices=("static", "fusion"),
        default=["static", "fusion"],
    )
    defaults = TrainConfig()
    train.add_argument("--size", nargs=3, type=int, default=defaults.size)
    for name in ("base_channels", "epochs", "patience"):
        train.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in (
        "static_dropout",
        "fusion_dropout",
        "learning_rate",
        "weight_decay",
        "flip_probability",
    ):
        train.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )
    train.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Optional seed for splitting and training; unset by default.",
    )
    train.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=defaults.amp
    )
    inference = commands.add_parser(
        "predict", help="Predict from a release-format checkpoint."
    )
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument(
        "--cases", type=Path, required=True, help="CSV with CaseID only."
    )
    for command in (train, inference):
        command.add_argument("--npy", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--device", choices=("auto", "cpu", "cuda"), default="auto"
        )
        command.add_argument("--workers", type=int, default=defaults.workers)
        command.add_argument("--threads", type=int, default=defaults.threads)
        command.add_argument("--batch-size", type=int, default=defaults.batch_size)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        require(len(args.models) == len(set(args.models)), "DUPLICATE_MODEL")
        run_training(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    raise SystemExit(safe_main(main))
