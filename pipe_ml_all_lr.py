"""Train and use the OpenFOAM axial-velocity surrogate in one Python file.

Edit the settings near the top, then run: python3 pipe_ml_with_cutoffs.py
Requires NumPy, pandas, and PyTorch. Each training and validation case folder
contains its own field.csv and case.json.
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn


# Settings

# 1. Choose train, predict, cutoff_report, steady_report, or raw_predict.
MODE = "train"

# 2. For training, list folders that each contain field.csv and case.json.
# Use absolute paths, paths starting with ~/, or paths relative to this script.
TRAIN_CASES = [
    "ml_data_D0.1,U0.001",
    "ml_data_D0.1,U0.002",
    "ml_data_D0.1,U0.003",
    "ml_data_D0.1,U0.004",
    "ml_data_D0.1,U0.005",
    "ml_data_D0.1,U0.006",
    "ml_data_D0.1,U0.007",
    "ml_data_D0.1,U0.009",
    "ml_data_D0.1,U0.010",
    "ml_data_D0.1,U0.011",
    "ml_data_D0.1,U0.012",
    "ml_data_D0.1,U0.013",
    "ml_data_D0.1,U0.016",
    "ml_data_D0.1,U0.018",
    "ml_data_D0.1,U0.019",
    "ml_data_D0.1,U0.02",
    "ml_data_D0.15,U0.001",
    "ml_data_D0.15,U0.002",
    "ml_data_D0.15,U0.003",
    "ml_data_D0.15,U0.005",
    "ml_data_D0.15,U0.006",
    "ml_data_D0.15,U0.007",
    "ml_data_D0.15,U0.008",
    "ml_data_D0.15,U0.010",
    "ml_data_D0.15,U0.011",
    "ml_data_D0.15,U0.012",
]
VALIDATION_CASES = [
    "ml_data_D0.15,U0.009",
    "ml_data_D0.1,U0.017",
    "ml_data_D0.1,U0.008",
    "ml_data_D0.15,U0.004",
]

# Training creates this folder. Prediction reads velocity.pt from it.
# Give each new training run a fresh output folder.
RUN_FOLDER = "runs/4x64lr"

# 3. Model and training settings.
TRAIN_SETTINGS = dict(
    layers=4,
    neurons=64,
    epochs=100,
    patience=25,
    batch_size=1024,
    condition_batch_size=256,
    sample_fraction=1.0,
    learning_rate=0.001,
    seed=42,
)
CPU_THREADS = 1

# Scheduler patience=3 means reduce after four epochs without improvement.
LR_SCHEDULE = dict(factor=0.5, patience=3, min_lr=1e-5)

# 4. Edit this block after training, when MODE is "predict".
# Lengths and positions are in metres, speeds in m/s, times in seconds.
PREDICTION = dict(
    diameter=0.15,
    length=19.0,
    speed=0.007,
    nu=1e-6,
    x=16.0,
    r=0.0,
    time=None,        # None uses the estimated steady time in predict mode.
    steady_time=None,  # Only used by raw_predict. A number overrides its scan.
)

STEADY_SCAN = dict(
    scan_until=None,
    time_step=50.0,
    steady_tolerance=0.1,
    steady_hold=200.0,
    axial_step=0.05,
    radial_points=30,
)


# OpenFOAM case loading and data preparation

VELOCITY_COLUMNS = {"x_m", "r_m", "t_s", "u_axial_m_s", "sample_type"}
MODEL_INPUTS = ("Re", "x_star", "r_star_squared", "t_star")


@dataclass
class PipeCase:
    case_id: str
    diameter: float
    length: float
    speed: float
    viscosity: float
    le_m: float | None
    steady_s: float | None
    steady_confirmed_s: float | None
    steady_status: str | None
    development_reference_s: float | None
    field: pd.DataFrame


def optional_nonnegative(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number or null.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return float(value)


def load_case(directory, *, csv_path=None, json_path=None):
    """Read the paired exports, preserving all CSV columns and case metadata."""
    directory = Path(directory)
    csv_path = Path(csv_path) if csv_path is not None else directory / "field.csv"
    json_path = Path(json_path) if json_path is not None else directory / "case.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    pipe = metadata["pipe"]
    diameter, length, speed, viscosity = (
        optional_nonnegative(pipe[key], f"pipe.{key}")
        for key in ("diameter", "length", "speed", "nu")
    )
    if not all(value is not None and value > 0 for value in (diameter, length, speed, viscosity)):
        raise ValueError(f"{json_path}: diameter, length, speed, and nu must be positive.")

    le_m = optional_nonnegative(metadata.get("development"), "development")
    if le_m is not None and not 0 < le_m <= length:
        raise ValueError(f"{json_path}: development length must be within the pipe.")

    field = pd.read_csv(csv_path)
    missing = VELOCITY_COLUMNS.difference(field.columns)
    if missing or field.empty:
        raise ValueError(f"{csv_path}: expected field rows with {sorted(VELOCITY_COLUMNS)}; missing {sorted(missing)}.")
    for column, maximum in (("x_m", length), ("r_m", diameter / 2), ("t_s", math.inf)):
        values = field[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < -1e-7).any() or (values > maximum + 1e-7).any():
            raise ValueError(f"{csv_path}: {column} is outside the physical range in {json_path}.")
    saved_times = metadata.get("times_s")
    if saved_times is not None:
        actual = np.sort(field["t_s"].unique())
        expected = np.sort(np.asarray(saved_times, dtype=float))
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-8, atol=1e-7):
            raise ValueError(f"{csv_path}: sampled times do not match {json_path}.")
    inlet = field.loc[field["sample_type"].eq("inlet"), "u_axial_m_s"]
    if not inlet.empty and not math.isclose(float(inlet.median()), speed, rel_tol=1e-4, abs_tol=1e-8):
        raise ValueError(f"{csv_path}: inlet velocity does not match pipe.speed in {json_path}.")

    return PipeCase(
        case_id=str(metadata.get("case_id") or directory.name),
        diameter=diameter,
        length=length,
        speed=speed,
        viscosity=viscosity,
        le_m=le_m,
        steady_s=optional_nonnegative(metadata.get("steady"), "steady"),
        steady_confirmed_s=optional_nonnegative(metadata.get("steady_confirmed_s"), "steady_confirmed_s"),
        steady_status=metadata.get("steady_status"),
        development_reference_s=optional_nonnegative(
            metadata.get("development_reference_time_s"), "development_reference_time_s"
        ),
        field=field,
    )


def axial_training_limit(case, le_factor=None):
    """Return the full length or the optional cap min(L, le_factor * Le)."""
    if le_factor is None:
        return case.length
    if not math.isfinite(le_factor) or le_factor < 1:
        raise ValueError("Use a finite Le factor of at least 1.")
    if case.le_m is None:
        raise ValueError(f"{case.case_id}: an Le crop requires a known development length.")
    return min(case.length, le_factor * case.le_m)


def prepare_velocity_groups(case, *, le_factor=None):
    """Build CFD, inlet, and initial groups using each case's physical parameters.

    Use the default full domain for validation. Apply le_factor to training only
    for a controlled crop experiment. Le and case dimensions stay available via
    PipeCase and are excluded from MODEL_INPUTS.
    """
    limit = axial_training_limit(case, le_factor)
    field = case.field.loc[case.field["x_m"] <= limit + 1e-7]
    x, r, time = field["x_m"], field["r_m"], field["t_s"]
    U, D = case.speed, case.diameter
    kind = field["sample_type"]

    rows = pd.DataFrame({
        "x_m": x,
        "Re": U * D / case.viscosity,
        "x_star": x / D,
        "r_star_squared": (r / (D / 2)) ** 2,
        "t_star": time * U / D,
    })
    rows["region"] = (
        "unknown" if case.le_m is None
        else np.where(x < case.le_m, "developing", "developed")
    )

    cfd = kind.isin(("interior", "axis")) & time.gt(0)
    inlet = kind.eq("inlet") & rows["r_star_squared"].lt(1)
    initial = kind.eq("interior") & time.eq(0)
    targets = field.loc[cfd, "u_axial_m_s"] / U
    if targets.empty or not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise ValueError(f"{case.case_id}: expected finite positive-time CFD velocity targets.")

    return {
        "cfd": rows.loc[cfd].assign(target=targets),
        "inlet": rows.loc[inlet].assign(target=1.0),
        "initial": rows.loc[initial].assign(target=0.0),
    }


# Velocity training and prediction

GROUPS = ("cfd", "inlet", "initial")


class VelocityNet(nn.Module):
    """Map dimensionless inputs to u_x/U with an exact no-slip wall factor."""

    def __init__(self, layers, neurons, lower, upper):
        super().__init__()
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.float32))
        span = self.upper - self.lower
        self.register_buffer("span", torch.where(span > 0, span, torch.ones_like(span)))
        blocks = []
        width = len(MODEL_INPUTS)
        for _ in range(layers):
            blocks.extend((nn.Linear(width, neurons), nn.Tanh()))
            width = neurons
        self.network = nn.Sequential(*blocks, nn.Linear(width, 1))

    def forward(self, inputs):
        scaled = 2 * (inputs - self.lower) / self.span - 1
        return (1 - inputs[:, 2]) * self.network(scaled).squeeze(-1)


def input_tensor(rows):
    return torch.from_numpy(rows[list(MODEL_INPUTS)].to_numpy(dtype=np.float32, copy=True))


@torch.inference_mode()
def infer(model, inputs, batch_size=8192):
    device = model.lower.device
    return np.concatenate([model(batch.to(device)).cpu().numpy() for batch in inputs.split(batch_size)])


def raw_velocity(model, U, D, nu, x, r, t):
    """Predict in m/s. x and r follow NumPy broadcasting rules."""
    x, r = np.broadcast_arrays(x, r)
    inputs = np.column_stack((
        np.full(x.size, U * D / nu),
        x.ravel() / D,
        (r.ravel() / (D / 2)) ** 2,
        np.full(x.size, t * U / D),
    ))
    predictions = infer(model, torch.as_tensor(inputs, dtype=torch.float32))
    return (U * predictions.astype(float)).reshape(x.shape)


def subsample_cfd(rows, fraction, rng):
    """Keep the requested fraction within each time and flow-region stratum."""
    if fraction == 1:
        return rows
    indices = []
    for positions in rows.groupby(["t_star", "region"], sort=False).indices.values():
        count = math.ceil(fraction * len(positions))
        indices.extend(rng.choice(positions, size=count, replace=False))
    return rows.iloc[indices]


def velocity_metrics(model, validation):
    """Report raw-prediction errors per case and per CFD reference region."""
    results = []
    for label, case, features, truth, regions in validation:
        error = (infer(model, features).astype(float) - truth) * case.speed
        for region in ("all", "developing", "developed", "unknown"):
            values = error if region == "all" else error[regions == region]
            if not len(values):
                continue
            results.append({
                "case": label, "diameter_m": case.diameter, "speed_m_s": case.speed,
                "region": region, "rows": len(values),
                "mae_m_s": float(np.abs(values).mean()),
                "rmse_m_s": float(np.sqrt(np.square(values).mean())),
            })
    return results


def validation_data(paths):
    cases = []
    for path in paths:
        case = load_case(path)
        rows = prepare_velocity_groups(case)["cfd"]
        cases.append((
            str(path), case, input_tensor(rows), rows["target"].to_numpy(dtype=float),
            rows["region"].to_numpy(),
        ))
    return cases


def write_results(path, records):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def report_steady_times(model, cases, directory):
    records = []
    for label, case in cases:
        U, D, nu, L = case.speed, case.diameter, case.viscosity, case.length
        predicted, scan_end = scan_steady_time(model, U, D, nu, L, SimpleNamespace(**STEADY_SCAN))
        cfd = case.steady_s
        records.append(dict(
            case=label,
            openfoam_steady_time_s=cfd,
            predicted_steady_time_s=predicted,
            scan_end_s=scan_end,
            absolute_error_s=None if cfd is None or predicted is None else abs(predicted - cfd),
        ))
        cfd_text = f"{cfd:g} s" if cfd is not None else "unavailable"
        ml_text = f"{predicted:g} s" if predicted is not None else f"not detected by {scan_end:g} s"
        print(f"{label} | OpenFOAM steady time: {cfd_text} | ML steady time: {ml_text}")
    output = directory / "steady_time_metrics.csv"
    write_results(output, records)
    print(f"Saved {output}")


def train(args):
    if min(args.layers, args.neurons, args.epochs, args.patience, args.batch_size,
           args.condition_batch_size) < 1 or not (0 < args.sample_fraction <= 1):
        raise ValueError("Use positive counts and a sample fraction in (0, 1].")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    paths = [path.resolve() for path in args.train + args.validation]
    if len(set(paths)) != len(paths):
        raise ValueError("Each case directory must appear only once in the split.")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    training = {name: [] for name in GROUPS}
    training_conditions = set()
    for path in args.train:
        case = load_case(path)
        training_conditions.add((case.diameter, case.speed, case.viscosity))
        groups = prepare_velocity_groups(case)
        groups["cfd"] = subsample_cfd(groups["cfd"], args.sample_fraction, rng)
        for name in GROUPS:
            rows = groups[name]
            training[name].append((input_tensor(rows), torch.tensor(rows["target"].to_numpy(), dtype=torch.float32)))

    validation = validation_data(args.validation)
    if any((case.diameter, case.speed, case.viscosity) in training_conditions
           for _, case, *_ in validation):
        raise ValueError("Keep each diameter/speed/viscosity combination on one side of the split.")

    data = {name: (torch.cat([item[0] for item in batches]), torch.cat([item[1] for item in batches]))
            for name, batches in training.items()}
    for name, (_, target) in data.items():
        if not len(target):
            raise ValueError(f"The training cases have zero rows in the {name} group.")
    lower = torch.stack([features.amin(dim=0) for features, _ in data.values()]).amin(dim=0)
    upper = torch.stack([features.amax(dim=0) for features, _ in data.values()]).amax(dim=0)
    device = torch.device(getattr(args, "device", "cpu"))
    model = VelocityNet(args.layers, args.neurons, lower, upper).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if LR_SCHEDULE["min_lr"] > args.learning_rate:
        raise ValueError("Scheduler min_lr must not exceed the initial learning rate.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", threshold=0.0, threshold_mode="abs", **LR_SCHEDULE,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "velocity.pt"
    cfd_count = len(data["cfd"][1])
    print(f"Training rows: { {name: len(target) for name, (_, target) in data.items()} }")

    history = []
    best_mae = math.inf
    best_epoch = updates = 0
    total_train_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = perf_counter()
        order = rng.permutation(cfd_count)
        for start in range(0, cfd_count, args.batch_size):
            indices = {"cfd": order[start:start + args.batch_size]}
            for name in ("inlet", "initial"):
                count = len(data[name][1])
                indices[name] = rng.choice(count, size=min(count, args.condition_batch_size), replace=False)
            losses = []
            for name, (features, targets) in data.items():
                batch = indices[name]
                inputs = features[batch].to(device)
                truth = targets[batch].to(device)
                losses.append((model(inputs) - truth).square().mean())
            optimizer.zero_grad(set_to_none=True)
            sum(losses).backward()
            optimizer.step()
            updates += 1

        elapsed = perf_counter() - started
        total_train_seconds += elapsed
        model.eval()
        metrics = velocity_metrics(model, validation)
        mae = float(np.mean([row["mae_m_s"] for row in metrics if row["region"] == "all"]))
        if not math.isfinite(mae):
            raise ValueError("Validation MAE became non-finite.")
        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step(mae)
        lr_next = optimizer.param_groups[0]["lr"]
        if lr_next < lr_used:
            print(f"Epoch {epoch}: learning rate {lr_used:.6g} -> {lr_next:.6g}", flush=True)
        history.append({
            "lr_used": lr_used, "lr_next": lr_next,
            "epoch": epoch, "optimizer_updates": updates, "validation_case_mean_mae_m_s": mae,
            "epoch_training_seconds": elapsed, "cumulative_training_seconds": total_train_seconds,
            "layers": args.layers, "neurons_per_layer": args.neurons,
            "cfd_samples_per_epoch": cfd_count, "sample_fraction": args.sample_fraction,
            "batch_size": args.batch_size, "condition_batch_size": args.condition_batch_size,
            "learning_rate": args.learning_rate, "seed": args.seed, "cpu_threads": args.threads,
        })
        write_results(args.output / "history.csv", history)
        print(f"Epoch {epoch}: case-mean MAE {mae:.7g} m/s | {elapsed:.2f} s training", flush=True)
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
            pending_checkpoint = checkpoint.with_suffix(".tmp")
            torch.save({
                "layers": args.layers, "neurons": args.neurons,
                "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "best": history[-1],
            }, pending_checkpoint)
            pending_checkpoint.replace(checkpoint)
        if epoch - best_epoch >= args.patience:
            break

    model, best = load_model(checkpoint)
    print(f"Best epoch {best_epoch}, update {best['optimizer_updates']} | {best_mae:.7g} m/s")
    write_results(args.output / "velocity_metrics.csv", velocity_metrics(model, validation))
    report_cutoffs(model, args.train, [(label, case) for label, case, *_ in validation], args.output)
    report_steady_times(model, [(label, case) for label, case, *_ in validation], args.output)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    model = VelocityNet(checkpoint["layers"], checkpoint["neurons"], state["lower"], state["upper"])
    model.load_state_dict(state)
    return model.eval(), checkpoint["best"]


def detect_steady_time(times, sample_velocity, speed, tolerance_percent, hold_seconds):
    """Find the earliest sampled time whose velocity field stays flat to scan end."""
    if len(times) < 2 or times[-1] - times[0] < hold_seconds - 1e-9:
        return None
    limit = 2 * speed * tolerance_percent / 100
    last = sample_velocity(float(times[-1]))
    if not np.isfinite(last).all():
        raise ValueError("The steady-time scan produced non-finite velocities.")
    low, high = last, last
    steady_time = None
    for time in times[-2::-1]:
        values = sample_velocity(float(time))
        if not np.isfinite(values).all():
            raise ValueError("The steady-time scan produced non-finite velocities.")
        low = np.minimum(low, values)
        high = np.maximum(high, values)
        if np.max(high - low) > limit:
            # Adding still earlier snapshots can only increase the spread.
            break
        if times[-1] - time >= hold_seconds - 1e-9:
            steady_time = float(time)
    return steady_time


def scan_steady_time(model, U, D, nu, L, settings):
    """Run the existing velocity plateau test for a specified pipe."""
    count = max(2, math.ceil(L / settings.axial_step))
    x = (np.arange(count) + 0.5) * L / count
    eta = (np.arange(settings.radial_points) + 0.5) / settings.radial_points
    radii = np.r_[0.0, eta * (D / 2)]
    low, high = model.lower.numpy(), model.upper.numpy()
    first_time, last_time = (float(bound[3]) * D / U for bound in (low, high))
    tolerance_s = max(1e-6, 1e-7 * last_time)
    scan_end = last_time if settings.scan_until is None else settings.scan_until
    if not first_time <= scan_end <= last_time + tolerance_s:
        raise ValueError(f"Scan end must be within the model time range [{first_time:g}, {last_time:g}] s.")
    scan_end = min(scan_end, last_time)
    if x[-1] / D > high[1] + 1e-6:
        print("Part of the requested axial scan is outside the training range.")
    times = np.r_[np.arange(first_time, scan_end, settings.time_step), scan_end]

    def monitor(time):
        return raw_velocity(model, U, D, nu, x[:, None], radii[None, :], time)

    steady_time = detect_steady_time(times, monitor, U, settings.steady_tolerance, settings.steady_hold)
    return steady_time, scan_end


def predict_with_raw_scan(args):
    model, _ = load_model(args.checkpoint)
    U, D, nu, L = args.speed, args.diameter, args.nu, args.length
    values = (U, D, nu, L, args.axial_step, args.x, args.r, args.steady_time,
              args.time, args.scan_until, args.time_step, args.steady_tolerance, args.steady_hold)
    if not all(value is None or math.isfinite(value) for value in values):
        raise ValueError("Supply finite geometry, time, and flow values.")
    if min(U, D, nu, L, args.axial_step) <= 0 or args.radial_points < 2:
        raise ValueError("Geometry, viscosity, speed and spacing must be positive with at least two radial points.")
    if not (0 <= args.x <= L and 0 <= args.r <= D / 2):
        raise ValueError("Use x within the pipe and r within the radius.")
    if min(args.time_step, args.steady_tolerance, args.steady_hold) <= 0:
        raise ValueError("Time step, steady tolerance and hold duration must be positive.")
    if args.time is not None and args.time < 0:
        raise ValueError("Use a nonnegative query time.")

    low, high = model.lower.numpy(), model.upper.numpy()
    first_time, last_time = (float(bound[3]) * D / U for bound in (low, high))
    tolerance_s = max(1e-6, 1e-7 * last_time)
    reynolds = U * D / nu
    if not (low[0] - 1e-6 <= reynolds <= high[0] + 1e-6):
        print("Requested Reynolds number is outside the training range.")

    steady_time = args.steady_time
    if steady_time is None:
        steady_time, scan_end = scan_steady_time(model, U, D, nu, L, args)
        if steady_time is None:
            print(f"Steady time not detected by {scan_end:g} s.")
        else:
            print(f"Predicted steady time: {steady_time:g} s (checked through {scan_end:g} s)")
    else:
        if not first_time <= steady_time <= last_time + tolerance_s:
            raise ValueError(f"Supplied steady time must be within [{first_time:g}, {last_time:g}] s.")
        steady_time = min(steady_time, last_time)
        print(f"Using supplied steady time: {steady_time:g} s")

    time = args.time
    if time is None:
        time = scan_end if steady_time is None else steady_time
    is_steady = steady_time is not None and time >= steady_time
    t_model = steady_time if is_steady else time
    if not first_time - tolerance_s <= t_model <= last_time + tolerance_s:
        raise ValueError(f"The time used to evaluate the model must be within [{first_time:g}, {last_time:g}] s.")
    t_model = min(max(t_model, first_time), last_time)
    x_model = args.x
    if not low[1] - 1e-6 <= x_model / D <= high[1] + 1e-6:
        print("The axial position used for the query is outside the training range.")
    velocity = float(raw_velocity(model, U, D, nu, x_model, args.r, t_model))
    print(f"Predicted u_x: {velocity:.8g} m/s (model evaluated at x={x_model:g} m, t={t_model:g} s)")


class CutoffPredictor:
    def __init__(self, case_dirs, model):
        self.le_labels = []
        self.time_labels = []
        seen = set()
        for folder in case_dirs:
            folder = Path(folder).expanduser().resolve()
            if folder in seen:
                continue
            seen.add(folder)
            path = folder / "case.json"
            meta = json.loads(path.read_text())
            pipe = meta["pipe"]
            D, L, U, nu = (float(pipe[key]) for key in ("diameter", "length", "speed", "nu"))
            if not np.isfinite([D, L, U, nu]).all() or min(D, L, U, nu) <= 0:
                raise ValueError(f"Invalid pipe conditions in {path}")
            re = U * D / nu
            le, steady = meta.get("development"), meta.get("steady")
            if le is not None:
                le = float(le)
                if not 0 < le <= L:
                    raise ValueError(f"Invalid development length in {path}")
                self.le_labels.append((re, le / D))
            if steady is not None:
                steady = float(steady)
                if not np.isfinite(steady) or steady < 0:
                    raise ValueError(f"Invalid steady time in {path}")
                self.time_labels.append((re, L / D, steady * U / D))

        if len(set(re for re, _ in self.le_labels)) < 2 or len(self.time_labels) < 2:
            raise ValueError("Need development labels at two Reynolds numbers and at least two steady labels.")

        self.re_grid, groups = np.unique(
            [re for re, _ in self.le_labels], return_inverse=True
        )
        lengths = np.array([le for _, le in self.le_labels])
        self.le_grid = np.array([np.mean(lengths[groups == i]) for i in range(len(self.re_grid))])

        times = np.array(self.time_labels)
        self.time_lower = times[:, :2].min(axis=0)
        self.time_upper = times[:, :2].max(axis=0)
        self.time_active = self.time_upper > self.time_lower
        self.time_span = np.where(self.time_active, self.time_upper - self.time_lower, 1.0)
        coordinates = ((times[:, :2] - self.time_lower) / self.time_span)[:, self.time_active]
        design = np.column_stack((np.ones(len(times)), coordinates))
        self.time_coefficients, _, rank, _ = np.linalg.lstsq(design, times[:, 2], rcond=None)
        if rank != design.shape[1] or not self.time_active[0]:
            raise ValueError("Steady labels need independent coverage of Reynolds number and any varying L/D.")
        self.model = model

    def cutoffs(self, D, L, U, nu):
        if not np.isfinite([D, L, U, nu]).all() or min(D, L, U, nu) <= 0:
            raise ValueError("D, L, U and ν must be finite and positive.")
        re = U * D / nu
        if not self.re_grid[0] <= re <= self.re_grid[-1]:
            raise ValueError(f"Development labels cover Re={self.re_grid[0]:g} to {self.re_grid[-1]:g}.")
        query = np.array([re, L / D])
        tolerance = 1e-7 * self.time_upper
        if np.any(query < self.time_lower - tolerance) or np.any(query > self.time_upper + tolerance):
            raise ValueError(
                f"Steady labels cover Re={self.time_lower[0]:g}–{self.time_upper[0]:g} "
                f"and L/D={self.time_lower[1]:g}–{self.time_upper[1]:g}."
            )
        normalized = ((query - self.time_lower) / self.time_span)[self.time_active]
        ts_star = float(np.r_[1.0, normalized] @ self.time_coefficients)
        if ts_star < 0 or not np.isfinite(ts_star):
            raise ValueError("Steady-time regression returned an invalid cutoff.")
        return D * np.interp(re, self.re_grid, self.le_grid), ts_star * D / U

    def predict(self, D, L, U, nu, x, r, t):
        le, ts = self.cutoffs(D, L, U, nu)
        r = np.asarray(r, dtype=float)
        if (not np.isfinite([x, t]).all() or not np.isfinite(r).all()
                or not 0 <= x <= L or t < 0 or np.any((r < 0) | (r > D / 2))):
            raise ValueError("Use 0 ≤ x ≤ L, 0 ≤ r ≤ D/2, and t ≥ 0.")
        # The entrance-length label describes steady flow.
        x_eval = min(x, le) if t >= ts else x
        t_eval = min(t, ts)
        values = np.array([U * D / nu, x_eval / D, t_eval * U / D])
        indices = [0, 1, 3]
        lo = self.model.lower.numpy()[indices]
        hi = self.model.upper.numpy()[indices]
        tolerance = 1e-6 * np.maximum(1, np.abs(hi))
        if np.any(values < lo - tolerance) or np.any(values > hi + tolerance):
            raise ValueError("The effective Re, x/D or tU/D is outside the velocity checkpoint's input range.")
        ux = raw_velocity(self.model, U, D, nu, x_eval, r, t_eval)
        return dict(ux=ux, le=le, steady_time=ts, x_eval=x_eval, t_eval=t_eval)


def show_prediction(result):
    print(f"Predicted Le: {result['le']:.6g} m")
    print(f"Predicted steady time: {result['steady_time']:.6g} s")
    print(f"ML evaluated at x={result['x_eval']:.6g} m, t={result['t_eval']:.6g} s")


def notebook():
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    run_folder, training, _ = configured_paths()
    model, _ = load_model(run_folder / "velocity.pt")
    predictor = CutoffPredictor(training, model)
    print(f"Cutoff labels: {len(predictor.le_labels)} for Le, {len(predictor.time_labels)} for steady time")

    def update(D, L, U, nu, x, r, t):
        radii = np.linspace(0, D / 2, 201)
        try:
            result = predictor.predict(D, L, U, nu, x, np.r_[r, radii], t)
        except ValueError as error:
            print(error)
            return
        ux, profile = result["ux"][0], result["ux"][1:]
        analytical = 2 * U * (1 - (radii / (D / 2)) ** 2)
        show_prediction(result)
        print(f"Axial velocity at r={r:g} m: {ux:.8g} m/s")
        if result["le"] > L:
            print("Predicted development length exceeds the pipe length.")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(radii / (D / 2), profile, label="ML prediction with cutoffs")
        ax.plot(radii / (D / 2), analytical, "--", label="Fully developed analytical reference")
        ax.scatter(r / (D / 2), ux, color="C0", zorder=3)
        ax.set(xlabel="Radius r/R", ylabel="Axial velocity (m/s)")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.show()
        plt.close(fig)

    widgets.interact_manual(
        update,
        D=widgets.FloatText(value=0.15, description="D [m]"),
        L=widgets.FloatText(value=19, description="L [m]"),
        U=widgets.FloatText(value=0.007, description="U [m/s]"),
        nu=widgets.FloatText(value=1e-6, description="ν [m²/s]"),
        x=widgets.FloatText(value=16, description="x [m]"),
        r=widgets.FloatText(value=0, description="r [m]"),
        t=widgets.FloatText(value=1820, description="t [s]"),
    )



def configured_paths():
    here = Path(__file__).resolve().parent
    run_folder = here / Path(RUN_FOLDER).expanduser()
    training = [here / Path(folder).expanduser() for folder in TRAIN_CASES]
    # Use these new cases once their exports are available for this grid.
    pending = {here / folder for folder in ['ml_data_D0.15,U0.012', 'ml_data_D0.1,U0.013']}
    for path in pending.intersection(training):
        if not all((path / name).is_file() for name in ("field.csv", "case.json")):
            print(f"Pending training export: {path}", flush=True)
            training.remove(path)
    validation = [here / Path(folder).expanduser() for folder in VALIDATION_CASES]
    return run_folder, training, validation


def report_cutoffs(model, training, cases, directory):
    try:
        predictor = CutoffPredictor(training, model)
    except (OSError, ValueError, KeyError) as error:
        print(f"Cutoff report unavailable: {error}")
        return
    records = []
    for label, case in cases:
        le = ts = None
        status = "ok"
        try:
            le, ts = predictor.cutoffs(case.diameter, case.length, case.speed, case.viscosity)
        except ValueError as error:
            status = str(error)
        records.append(dict(
            case=label, openfoam_le_m=case.le_m, predicted_le_m=le,
            le_absolute_error_m=None if le is None or case.le_m is None else abs(le - case.le_m),
            openfoam_steady_time_s=case.steady_s, predicted_steady_time_s=ts,
            steady_absolute_error_s=None if ts is None or case.steady_s is None else abs(ts - case.steady_s),
            status=status,
        ))
        if status == "ok":
            print(f"{label} | estimated Le: {le:g} m | estimated steady time: {ts:g} s")
        else:
            print(f"{label} | {status}")
    output = directory / "cutoff_metrics.csv"
    write_results(output, records)
    print(f"Saved {output}")


def predict_with_cutoffs(model, training):
    settings = SimpleNamespace(**PREDICTION)
    predictor = CutoffPredictor(training, model)
    conditions = (settings.diameter, settings.length, settings.speed, settings.nu)
    time = settings.time
    if time is None:
        _, time = predictor.cutoffs(*conditions)
    result = predictor.predict(*conditions, settings.x, settings.r, time)
    show_prediction(result)
    if result["le"] > settings.length:
        print("Predicted development length exceeds the pipe length.")
    print(f"Predicted u_x: {float(result['ux']):.8g} m/s")


# Run the selected mode

def main():
    run_folder, training, validation = configured_paths()
    if CPU_THREADS < 1:
        raise ValueError("CPU_THREADS must be positive.")
    torch.set_num_threads(CPU_THREADS)

    if MODE == "train":
        if not TRAIN_CASES or not VALIDATION_CASES:
            raise ValueError("Fill in both TRAIN_CASES and VALIDATION_CASES.")
        train(SimpleNamespace(
            train=training,
            validation=validation,
            output=run_folder,
            threads=CPU_THREADS,
            **TRAIN_SETTINGS,
        ))
    elif MODE == "steady_report":
        if not VALIDATION_CASES:
            raise ValueError("Fill in VALIDATION_CASES.")
        model, _ = load_model(run_folder / "velocity.pt")
        report_steady_times(model, [(str(path), load_case(path)) for path in validation], run_folder)
    elif MODE in ("predict", "cutoff_report"):
        model, _ = load_model(run_folder / "velocity.pt")
        if MODE == "predict":
            predict_with_cutoffs(model, training)
        else:
            if not validation:
                raise ValueError("Fill in VALIDATION_CASES.")
            report_cutoffs(model, training, [(str(path), load_case(path)) for path in validation], run_folder)
    elif MODE == "raw_predict":
        checkpoint = run_folder / "velocity.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing at {checkpoint}. Train a model first or select its RUN_FOLDER.")
        predict_with_raw_scan(SimpleNamespace(
            checkpoint=checkpoint,
            **PREDICTION,
            **STEADY_SCAN,
        ))
    else:
        raise ValueError('Set MODE to train, predict, cutoff_report, steady_report, or raw_predict.')


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
