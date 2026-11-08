"""Create a submission using the validated baseline and R=50 specialist."""

import argparse
import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from train_tcn import (
    ARCHITECTURE, CONTROL_FEATURES, FEATURES, PROJECT, STEPS_PER_BREATH,
    VentilatorTCN, add_control_features,
)


@torch.inference_mode()
def predict(inputs, checkpoint, device, batch_size=256):
    names = tuple(checkpoint["features"])
    if names == FEATURES + CONTROL_FEATURES:
        inputs = add_control_features(inputs)
    elif names != FEATURES:
        raise ValueError("Unsupported checkpoint features.")
    if checkpoint.get("run", {}).get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match this code.")
    mean = checkpoint["feature_mean"].numpy()
    scale = checkpoint["feature_scale"].numpy()
    values = np.ascontiguousarray(((inputs - mean) / scale).transpose(0, 2, 1))
    if not np.isfinite(values).all():
        raise ValueError("Preprocessed inputs must be finite.")
    model = VentilatorTCN(len(names)).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    result = np.empty((len(inputs), STEPS_PER_BREATH), dtype=np.float32)
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(values[start:start + batch_size]).to(device)
        result[start:start + batch_size] = model(batch).cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    archive = PROJECT / "data/ventilator-pressure-prediction.zip"
    specialist_path = PROJECT / "outputs/tcn-r50-specialist-30/best.pt"
    manifest = json.loads(specialist_path.with_name("run.json").read_text())
    if manifest["result"]["selected_route"] != "baseline_plus_specialist":
        raise ValueError("The specialist was not selected in the validated run.")
    base_path = PROJECT / "outputs/tcn-control-cosine-60/best.pt"
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    specialist = torch.load(specialist_path, map_location="cpu", weights_only=True)
    if specialist["run"]["routing"]["base_checkpoint"] != str(base_path):
        raise ValueError("The specialist refers to a different baseline.")
    with archive.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != manifest["archive_sha256"]:
            raise ValueError("Data archive differs from the validated run.")
    with ZipFile(archive) as source:
        with source.open("test.csv") as file:
            rows = pd.read_csv(file, dtype={name: np.float32 for name in FEATURES})
        with source.open("sample_submission.csv") as file:
            sample = pd.read_csv(file)
    if list(sample.columns) != ["id", "pressure"]:
        raise ValueError("Unexpected submission schema.")
    if rows.empty or rows[["id", "breath_id"]].isna().any().any():
        raise ValueError("Test rows must have IDs and breath IDs.")
    if not rows.id.is_unique or not sample.id.is_unique:
        raise ValueError("Submission and test IDs must be unique.")
    if len(sample) != len(rows) or set(sample.id) != set(rows.id):
        raise ValueError("Sample and test IDs do not match.")
    rows = rows.sort_values(["breath_id", "time_step"])
    if not rows.groupby("breath_id").size().eq(STEPS_PER_BREATH).all():
        raise ValueError("Every test breath must have 80 samples.")
    inputs = rows[list(FEATURES)].to_numpy().reshape(-1, STEPS_PER_BREATH, len(FEATURES))
    if not np.isfinite(inputs).all():
        raise ValueError("Test inputs must be finite.")
    if not (inputs[:, :, -2:] == inputs[:, :1, -2:]).all():
        raise ValueError("R and C must remain constant within each breath.")
    if not np.isin(inputs[:, :, 2], [0, 1]).all():
        raise ValueError("u_out must be binary.")
    route = inputs[:, 0, FEATURES.index("R")] == 50
    prediction = np.empty(inputs.shape[:2], dtype=np.float32)
    for selected, checkpoint in ((~route, base), (route, specialist)):
        if selected.any():
            prediction[selected] = predict(inputs[selected], checkpoint, device)
    by_id = pd.Series(prediction.ravel(), index=rows.id)
    sample["pressure"] = sample.id.map(by_id)
    if not np.isfinite(sample.pressure).all():
        raise ValueError("All predictions must be finite.")
    output = PROJECT / "predictions.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    sample.to_csv(output, index=False)
    saved = pd.read_csv(output)
    np.testing.assert_array_equal(saved.id, sample.id)
    np.testing.assert_allclose(saved.pressure, sample.pressure, rtol=1e-6)
    assert list(saved.columns) == ["id", "pressure"] and np.isfinite(saved.pressure).all()
    print(f"Saved and verified {len(saved):,} predictions for {len(inputs):,} breaths.")
    print(f"Specialist breaths: {int(route.sum()):,}; baseline breaths: {int((~route).sum()):,}.")
    print(output)


if __name__ == "__main__":
    main()
