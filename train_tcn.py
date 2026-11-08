"""Train a temporal convolutional network on the PA3 ventilator data."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parent
FEATURES = ("time_step", "u_in", "u_out", "R", "C")
CONTROL_FEATURES = ("delta_time", "delta_u_in", "cumulative_u_in_time")
STEPS_PER_BREATH = 80
DILATIONS = (1, 2, 4, 8, 16, 32)
HIDDEN_CHANNELS = 64
DEFAULT_SPLIT_SEED = 102
ARCHITECTURE = {
    "name": "VentilatorTCN", "hidden_channels": HIDDEN_CHANNELS,
    "dilations": list(DILATIONS), "kernel_size": 3, "activation": "GELU",
    "padding": "symmetric", "residual": True,
}


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        # Pad both ends. The test CSV supplies the full control sequence.
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class VentilatorTCN(nn.Module):
    def __init__(self, input_channels=len(FEATURES)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, HIDDEN_CHANNELS, 1),
            *(TemporalBlock(HIDDEN_CHANNELS, dilation) for dilation in DILATIONS),
            nn.Conv1d(HIDDEN_CHANNELS, 1, 1),
        )

    def forward(self, features):
        # Input shape: (breaths, features, time steps).
        return self.network(features).squeeze(1)


def load_training_data(archive):
    columns = ("breath_id", *FEATURES, "pressure")
    with ZipFile(archive) as source, source.open("train.csv") as file:
        rows = pd.read_csv(
            file,
            usecols=columns,
            dtype={column: np.float32 for column in (*FEATURES, "pressure")},
        )

    rows.sort_values(["breath_id", "time_step"], inplace=True)
    if rows.empty or rows["breath_id"].isna().any():
        raise ValueError("Training rows must have breath IDs.")
    if not rows.groupby("breath_id").size().eq(STEPS_PER_BREATH).all():
        raise ValueError(f"Each breath must have {STEPS_PER_BREATH} time steps.")

    breath_settings = rows.iloc[::STEPS_PER_BREATH][["breath_id", "R", "C"]].reset_index(drop=True)
    inputs = rows.loc[:, list(FEATURES)].to_numpy().reshape(
        -1, STEPS_PER_BREATH, len(FEATURES)
    )
    pressure = rows["pressure"].to_numpy(copy=True).reshape(-1, STEPS_PER_BREATH)
    if not np.isfinite(inputs).all() or not np.isfinite(pressure).all():
        raise ValueError("Training inputs and pressures must be finite.")
    valve = inputs[:, :, FEATURES.index("u_out")]
    if not ((valve == 0) | (valve == 1)).all():
        raise ValueError("u_out must be 0 or 1.")
    lung = inputs[:, :, -2:]
    if not (lung == lung[:, :1]).all():
        raise ValueError("R and C must stay constant within each breath.")
    scored = valve == 0
    return inputs, pressure, scored, breath_settings


def add_control_features(inputs):
    time, opening = inputs[:, :, 0], inputs[:, :, 1]
    # Start each breath with zero change and zero accumulated valve opening.
    delta_time = np.diff(time, axis=1, prepend=time[:, :1])
    delta_opening = np.diff(opening, axis=1, prepend=opening[:, :1])
    # Approximate the area under the valve-opening curve between samples.
    mean_opening = opening - 0.5 * delta_opening
    cumulative_opening = np.cumsum(mean_opening * delta_time, axis=1)
    extra = np.stack((delta_time, delta_opening, cumulative_opening), axis=-1)
    return np.concatenate((inputs, extra), axis=-1)


def split_breaths(settings, seed):
    rng = np.random.default_rng(seed)
    validation = np.concatenate([
        rng.choice(indices, size=round(0.2 * len(indices)), replace=False)
        for indices in settings.groupby(["R", "C"]).indices.values()
    ])
    validation.sort()
    training = np.setdiff1d(np.arange(len(settings)), validation)
    return training, validation


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def train_epoch(model, optimizer, data, order, batch_size):
    model.train()
    for batch in order.split(batch_size):
        features, pressure, scored = (tensor.index_select(0, batch) for tensor in data)
        optimizer.zero_grad(set_to_none=True)
        errors = (model(features) - pressure).abs() * scored
        loss = errors.sum() / scored.sum(dtype=torch.float32).clamp_min(1)
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def validation_error(model, data, order, batch_size):
    model.eval()
    total_error = torch.zeros((), device=order.device)
    for batch in order.split(batch_size):
        features, pressure, scored = (tensor.index_select(0, batch) for tensor in data)
        total_error += ((model(features) - pressure).abs() * scored).sum()
    return total_error.item()


def write_manifest(path, run):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(run, indent=2) + "\n")
    temporary.replace(path)


def split_identifiers(settings, training, validation):
    ids = settings["breath_id"].to_numpy(dtype="<i8")
    train_ids, valid_ids = ids[training], ids[validation]
    digest = hashlib.sha256()
    for part in (train_ids, valid_ids):
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part.tobytes())
    return train_ids, valid_ids, digest.hexdigest()


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=PROJECT / "data/ventilator-pressure-prediction.zip"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=102, help="Seed for model initialization and training order.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--specialist-base", type=Path,
        help="Fine-tune on R=50 from this checkpoint and use the original model for other settings.",
    )
    parser.add_argument(
        "--control-features", action="store_true",
        help="Add time gaps, inlet valve changes, and cumulative inlet opening.",
    )
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="constant",
        help="Use cosine decay to lower the learning rate to one tenth of its starting value.",
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or min(args.seed, args.split_seed) < 0:
        parser.error("Epochs, batch size, and patience must be positive. Seeds must be nonnegative.")
    specialist = args.specialist_base is not None
    initial_lr = args.learning_rate if args.learning_rate is not None else (1e-4 if specialist else 1e-3)
    if not math.isfinite(initial_lr) or initial_lr <= 0:
        parser.error("Learning rate must be finite and positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable. Use an MPS-enabled PyTorch or pass --device cpu.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed)
    output = args.output or PROJECT / "outputs" / datetime.now().strftime("tcn-%Y%m%d-%H%M%S-%f")
    print(f"Loading {args.archive} on {device} with PyTorch {torch.__version__}", flush=True)
    inputs, pressure, scored, settings = load_training_data(args.archive)
    train, valid = split_breaths(settings, args.split_seed)
    train_ids, valid_ids, split_hash = split_identifiers(settings, train, valid)
    with args.archive.open("rb") as archive:
        archive_hash = hashlib.file_digest(archive, "sha256").hexdigest()

    base = None
    feature_names = FEATURES + CONTROL_FEATURES if args.control_features else FEATURES
    if specialist:
        base = torch.load(args.specialist_base, map_location="cpu", weights_only=True)
        base_run = base.get("run", {})
        base_split = base_run.get("split", {})
        if (base_split.get("seed", base["seed"]) != args.split_seed
                or base_split.get("sha256", split_hash) != split_hash
                or base_run.get("archive_sha256", archive_hash) != archive_hash):
            parser.error("The specialist must use the same data and split as its base checkpoint.")
        if base_run.get("routing") is not None or base_run.get("architecture", ARCHITECTURE) != ARCHITECTURE:
            parser.error("Use a general TCN checkpoint with the current architecture as the base.")
        feature_names = tuple(base["features"])
        if feature_names not in (FEATURES, FEATURES + CONTROL_FEATURES):
            parser.error("The base checkpoint uses an unsupported set of features.")
        if args.control_features and feature_names == FEATURES:
            parser.error("The specialist must reuse the base checkpoint's features.")

    if feature_names == FEATURES + CONTROL_FEATURES:
        inputs = add_control_features(inputs)
    if base is None:
        # Calculate feature scaling from the training breaths only.
        train_inputs = inputs[train]
        mean = train_inputs.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scale = train_inputs.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scale[scale == 0] = 1
        del train_inputs
    else:
        mean, scale = base["feature_mean"].numpy(), base["feature_scale"].numpy()

    active_valid = valid
    other_valid = np.empty(0, dtype=np.int64)
    if specialist:
        high_resistance = settings["R"].to_numpy() == 50
        train = train[high_resistance[train]]
        active_valid = valid[high_resistance[valid]]
        other_valid = valid[~high_resistance[valid]]
    train_scored_steps = int(scored[train].sum())
    active_scored_steps = int(scored[active_valid].sum())
    scored_steps = int(scored[valid].sum())
    if min(train_scored_steps, active_scored_steps) == 0:
        raise ValueError("Both training and validation must contain scored rows.")

    inputs = np.ascontiguousarray(((inputs - mean) / scale).transpose(0, 2, 1))
    if not np.isfinite(inputs).all():
        raise ValueError("Scaled inputs must be finite.")
    data = tuple(torch.from_numpy(array).to(device) for array in (inputs, pressure, scored))
    train_order = torch.from_numpy(train)
    valid_order = torch.from_numpy(active_valid).to(device)
    model = VentilatorTCN(len(feature_names)).to(device)
    other_error, baseline_mae = 0.0, None
    if base is not None:
        model.load_state_dict(base["model"])
        other_order = torch.from_numpy(other_valid).to(device)
        other_error = validation_error(model, data, other_order, args.batch_size)
        baseline_error = validation_error(model, data, valid_order, args.batch_size)
        baseline_mae = (baseline_error + other_error) / scored_steps
        print(f"Base validation MAE: {baseline_mae:.6f} cmH2O", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    min_lr = initial_lr / 10
    scheduler = None
    if args.lr_schedule == "cosine" and args.epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - 1, eta_min=min_lr
        )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best.pt"
    split_file = output / "split.npz"
    np.savez_compressed(
        split_file, train_breath_ids=train_ids, validation_breath_ids=valid_ids,
        fitted_breath_ids=settings.iloc[train]["breath_id"].to_numpy(dtype="<i8"),
    )
    run = {
        "archive": str(args.archive.resolve()), "archive_sha256": archive_hash,
        "split": {"seed": args.split_seed, "validation_fraction": 0.2,
                  "stratified_by": ["R", "C"], "sha256": split_hash, "file": split_file.name},
        "architecture": ARCHITECTURE,
        "preprocessing": {"steps_per_breath": STEPS_PER_BREATH, "features": list(feature_names),
                          "sort_order": ["breath_id", "time_step"], "dtype": "float32",
                          "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
                          "scaling_source": "base_checkpoint" if specialist else "training_breaths",
                          "scored_rows": "u_out == 0"},
        "training": {"seed": args.seed, "batch_size": args.batch_size, "epochs": args.epochs,
                     "patience": args.patience, "optimizer": "AdamW", "weight_decay": 0.01,
                     "loss": "masked_mae", "learning_rate": initial_lr,
                     "lr_schedule": args.lr_schedule,
                     "min_learning_rate": min_lr if scheduler is not None else initial_lr,
                     "breaths": len(train), "scored_rows": train_scored_steps},
        "validation": {"breaths": len(valid), "scored_rows": scored_steps,
                       "active_breaths": len(active_valid), "active_scored_rows": active_scored_steps},
        "routing": {"base_checkpoint": str(args.specialist_base.resolve()), "specialist_resistance": 50,
                    "baseline_validation_mae": baseline_mae} if specialist else None,
        "device": args.device, "pytorch_version": str(torch.__version__),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    manifest = output / "run.json"
    write_manifest(manifest, run)
    print(f"Training breaths: {len(train):,} | validation breaths: {len(valid):,} | "
          f"scored validation rows: {scored_steps:,}", flush=True)
    print(f"Features: {', '.join(feature_names)}", flush=True)
    print(f"Saving results in {output.resolve()}", flush=True)

    best_mae = math.inf
    best_epoch = 0
    with (output / "metrics.csv").open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow((
            "epoch", "train_seconds", "validation_seconds", "epoch_seconds",
            "validation_mae", "learning_rate", "active_validation_mae",
        ))
        log.flush()
        for epoch in range(1, args.epochs + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            synchronize(device)
            start = perf_counter()
            order = train_order[torch.randperm(len(train_order), generator=shuffle)].to(device)
            train_epoch(model, optimizer, data, order, args.batch_size)
            synchronize(device)
            train_end = perf_counter()
            active_error = validation_error(model, data, valid_order, args.batch_size)
            # Pool absolute errors over the routed validation rows.
            mae = (active_error + other_error) / scored_steps
            synchronize(device)
            end = perf_counter()

            if not math.isfinite(mae):
                raise RuntimeError(f"Validation MAE became non-finite in epoch {epoch}.")
            train_seconds = train_end - start
            val_seconds = end - train_end
            elapsed = end - start
            writer.writerow((epoch, f"{train_seconds:.3f}", f"{val_seconds:.3f}",
                             f"{elapsed:.3f}", f"{mae:.8f}", f"{learning_rate:.8g}",
                             f"{active_error / active_scored_steps:.8f}"))
            log.flush()
            print(f"Epoch {epoch:02d} | {elapsed:.2f} s total "
                  f"({train_seconds:.2f} s train + {val_seconds:.2f} s validation) | "
                  f"validation MAE {mae:.6f} cmH2O | LR {learning_rate:.3g}", flush=True)

            if mae < best_mae:
                best_mae, best_epoch = mae, epoch
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({
                    "model": {name: value.cpu() for name, value in model.state_dict().items()},
                    "features": feature_names,
                    "feature_mean": torch.from_numpy(mean),
                    "feature_scale": torch.from_numpy(scale),
                    "epoch": best_epoch,
                    "validation_mae": best_mae,
                    "seed": args.seed,
                    "lr_schedule": args.lr_schedule,
                    "run": run,
                }, temporary)
                temporary.replace(checkpoint)
            if epoch - best_epoch >= args.patience:
                print(f"Early stop after {args.patience} epochs without improvement.")
                break
            if scheduler is not None and epoch < args.epochs:
                scheduler.step()

    use_specialist = specialist and best_mae < baseline_mae
    run["result"] = {
        "epochs_completed": epoch, "best_epoch": best_epoch, "best_validation_mae": best_mae,
        "elapsed_seconds": perf_counter() - started, "checkpoint": checkpoint.name,
        "selected_route": ("baseline_plus_specialist" if use_specialist else "baseline")
        if specialist else "single_model",
    }
    write_manifest(manifest, run)
    print(f"Best validation MAE: {best_mae:.6f} cmH2O at epoch {best_epoch}.")
    if specialist:
        print(f"Selected route: {run['result']['selected_route']}.")


if __name__ == "__main__":
    main()
