"""
CORE COMPONENT 2
================
Controlled model-capacity experiments for the existing from-scratch CuPy ViT.

IMPORTANT
---------
This file DOES NOT modify anything inside model/.
It imports the exact same VisionTransformer used by core_component1.py and
changes only constructor hyperparameters for controlled capacity experiments.

The following are kept fixed across experiments:
    - dataset and train/validation split
    - image size and patch size
    - batch size
    - augmentation policy
    - RGB normalization
    - dropout rates
    - label smoothing
    - optimizer and AdamW hyperparameters
    - learning-rate schedule
    - random seed
    - number of training epochs

Capacity sweeps:
    1. Number of Transformer blocks
    2. Number of attention heads
    3. Embedding dimension
    4. MLP ratio

For every experiment this file reports/saves:
    - exact trainable parameter count
    - theoretical forward MACs
    - theoretical forward FLOPs (= 2 * MACs for multiply-add operations)
    - epoch-wise training loss/accuracy
    - epoch-wise validation loss/accuracy
    - best validation accuracy / epoch
    - inference latency (batch size 1)
    - inference throughput (images/second)
    - plots and CSV files

Expected project layout
-----------------------
project/
    core_component1.py
    core_component2.py      <-- this file
    model/                  <-- UNCHANGED
    train/                  <-- same dataset used by core_component1
"""

from pathlib import Path
import csv
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import cupy as cp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the COMPLETE existing training/data pipeline.
# Importing core_component1 does not call train() because its train call is
# protected by: if __name__ == '__main__':
import core_component_01 as core1

# Exact same model implementation. Nothing in model/ is changed.
from model.vit_architecture import VisionTransformer


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "core_component2_results"
HISTORY_DIR = RESULTS_DIR / "history"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"

for directory in (RESULTS_DIR, HISTORY_DIR, PLOTS_DIR, CHECKPOINT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PERSISTENT RUN LOGGING / PROGRESS
# ============================================================================

class TeeStream:
    """Write every terminal message to both terminal and a persistent log file."""

    def __init__(self, terminal_stream, log_file):
        self.terminal_stream = terminal_stream
        self.log_file = log_file

    def write(self, message):
        self.terminal_stream.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.terminal_stream, "isatty", lambda: False)()


def start_persistent_logging():
    """
    Start a new run log.

    All print() output after this point is saved to:
        core_component2_results/logs/run_YYYYMMDD_HHMMSS.log
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{timestamp}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    return log_path, log_file, original_stdout, original_stderr


def stop_persistent_logging(log_file, original_stdout, original_stderr):
    """Restore normal terminal streams and close the persistent log."""

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()


def save_progress(status, **extra):
    """
    Save the latest run state atomically enough for ordinary experiment use.

    This file is updated throughout training, so after the terminal closes you
    can see which experiment/epoch was last completed.
    """

    progress = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    progress.update(extra)

    path = RESULTS_DIR / "progress.json"
    temp_path = RESULTS_DIR / "progress.json.tmp"

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(progress, file, indent=2)
        file.flush()

    temp_path.replace(path)
    return path


# ============================================================================
# CONTROLLED EXPERIMENT SETTINGS
# ============================================================================

# Baseline = exactly the architecture configured in core_component1.py.
BASELINE = {
    "num_blocks": core1.NUM_BLOCKS,
    "num_heads": core1.NUM_HEADS,
    "embed_dim": core1.EMBED_DIM,
    "mlp_ratio": core1.MLP_RATIO,
}

# Edit these lists if you want fewer/more runs.
LAYER_VALUES = [2, 3, 4, 6]
HEAD_VALUES = [2, 4, 8, 16]
EMBED_DIM_VALUES = [128, 192, 256, 384]
MLP_RATIO_VALUES = [2, 4, 6]

# Which sweeps to execute.
RUN_LAYER_SWEEP = True
RUN_HEAD_SWEEP = True
RUN_EMBED_DIM_SWEEP = True
RUN_MLP_RATIO_SWEEP = True

# A controlled capacity comparison is clearer when every architecture receives
# the same optimization budget. Therefore Core Component 2 deliberately uses
# the same fixed number of epochs for every run rather than stopping different
# models at different epochs.
EXPERIMENT_EPOCHS = core1.EPOCHS

# GPU benchmark settings.
LATENCY_WARMUP_RUNS = 10
LATENCY_MEASURE_RUNS = 50
THROUGHPUT_BATCH_SIZE = core1.BATCH_SIZE


# ============================================================================
# SMALL HELPERS
# ============================================================================

def reset_random_seeds():
    """Reset all RNGs before constructing/training every model."""

    random.seed(core1.SEED)
    np.random.seed(core1.SEED)
    cp.random.seed(core1.SEED)


def experiment_name(prefix, value):
    return f"{prefix}_{value}"


def make_experiment_list():
    """
    Build controlled experiments.

    Only ONE architectural variable changes from BASELINE in each sweep.
    The baseline itself is trained only once and reused in all comparisons.
    """

    experiments = []

    baseline_cfg = dict(BASELINE)
    baseline_cfg.update({"name": "baseline", "sweep": "baseline", "sweep_value": 0})
    experiments.append(baseline_cfg)

    if RUN_LAYER_SWEEP:
        for value in LAYER_VALUES:
            if value == BASELINE["num_blocks"]:
                continue
            cfg = dict(BASELINE)
            cfg["num_blocks"] = value
            cfg.update({"name": experiment_name("layers", value), "sweep": "layers", "sweep_value": value})
            experiments.append(cfg)

    if RUN_HEAD_SWEEP:
        for value in HEAD_VALUES:
            if value == BASELINE["num_heads"]:
                continue
            if BASELINE["embed_dim"] % value != 0:
                raise ValueError(f"HEAD_VALUES contains {value}, but embed_dim {BASELINE['embed_dim']} is not divisible by it.")
            cfg = dict(BASELINE)
            cfg["num_heads"] = value
            cfg.update({"name": experiment_name("heads", value), "sweep": "heads", "sweep_value": value})
            experiments.append(cfg)

    if RUN_EMBED_DIM_SWEEP:
        for value in EMBED_DIM_VALUES:
            if value == BASELINE["embed_dim"]:
                continue
            if value % BASELINE["num_heads"] != 0:
                raise ValueError(f"EMBED_DIM_VALUES contains {value}, but it is not divisible by num_heads={BASELINE['num_heads']}.")
            cfg = dict(BASELINE)
            cfg["embed_dim"] = value
            cfg.update({"name": experiment_name("embed", value), "sweep": "embed_dim", "sweep_value": value})
            experiments.append(cfg)

    if RUN_MLP_RATIO_SWEEP:
        for value in MLP_RATIO_VALUES:
            if value == BASELINE["mlp_ratio"]:
                continue
            cfg = dict(BASELINE)
            cfg["mlp_ratio"] = value
            cfg.update({"name": experiment_name("mlp_ratio", value), "sweep": "mlp_ratio", "sweep_value": value})
            experiments.append(cfg)

    return experiments


# ============================================================================
# MODEL CREATION -- EXISTING MODEL CODE IS NOT TOUCHED
# ============================================================================

def build_model(num_classes, config):
    """Instantiate the same VisionTransformer class with one test config."""

    return VisionTransformer(
        num_classes=num_classes,
        image_size=core1.IMAGE_SIZE,
        patch_size=core1.PATCH_SIZE,
        in_channels=core1.CHANNELS,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_blocks=config["num_blocks"],
        mlp_ratio=config["mlp_ratio"],
        eps=1e-5,
        dropout_rate=core1.DROPOUT_RATE,
        embedding_dropout_rate=core1.EMBEDDING_DROPOUT_RATE,
    )


# ============================================================================
# EXACT PARAMETER COUNT
# ============================================================================

def count_parameters(model):
    """Count every trainable scalar directly from the model state_dict."""

    breakdown = {}
    total = 0

    for name, parameter in model.state_dict().items():
        count = int(parameter.size)
        breakdown[name] = count
        total += count

    return total, breakdown


# ============================================================================
# MACs / FLOPs
# ============================================================================

def calculate_macs_flops(num_classes, config):
    """
    Theoretical FORWARD-PASS matrix multiplication / linear-layer compute.

    MAC convention:
        one multiply + accumulate = 1 MAC

    FLOP convention:
        one multiplication = 1 FLOP
        one addition       = 1 FLOP
        therefore 1 MAC ~= 2 FLOPs

    Counted:
        - patch projection
        - Q, K, V projections
        - QK^T attention score multiplication
        - attention @ V multiplication
        - attention output projection Wo
        - both MLP linear layers
        - classifier

    Not included in MAC count:
        - LayerNorm elementwise arithmetic
        - GELU elementwise arithmetic
        - softmax exp/division
        - residual additions
        - dropout masking
        - positional additions

    This makes the reported MACs reproducible and directly comparable across
    the controlled architectural sweeps.
    """

    image_size = core1.IMAGE_SIZE
    patch_size = core1.PATCH_SIZE
    channels = core1.CHANNELS

    D = int(config["embed_dim"])
    H = int(config["num_heads"])
    L = int(config["num_blocks"])
    R = int(config["mlp_ratio"])

    if D % H != 0:
        raise ValueError(f"embed_dim={D} must be divisible by num_heads={H}")

    head_dim = D // H
    patches_per_side = image_size // patch_size
    num_patches = patches_per_side * patches_per_side
    num_tokens = num_patches + 1
    patch_dim = patch_size * patch_size * channels
    mlp_hidden = R * D

    # Patch projection: every patch (patch_dim) -> D.
    patch_projection = num_patches * patch_dim * D

    # Per Transformer block.
    qkv_projection = 3 * num_tokens * D * D

    # Written with H and head_dim on purpose to show how heads contribute.
    attention_scores = H * num_tokens * num_tokens * head_dim
    attention_values = H * num_tokens * num_tokens * head_dim

    output_projection = num_tokens * D * D

    mlp_first = num_tokens * D * mlp_hidden
    mlp_second = num_tokens * mlp_hidden * D

    per_block = (qkv_projection + attention_scores + attention_values + output_projection + mlp_first + mlp_second)

    transformer_total = L * per_block
    classifier = D * num_classes

    total_macs = patch_projection + transformer_total + classifier
    total_flops = 2 * total_macs

    details = {
        "num_patches": num_patches,
        "num_tokens": num_tokens,
        "head_dim": head_dim,
        "mlp_hidden": mlp_hidden,
        "patch_projection_macs": patch_projection,
        "qkv_macs_per_block": qkv_projection,
        "attention_score_macs_per_block": attention_scores,
        "attention_value_macs_per_block": attention_values,
        "output_projection_macs_per_block": output_projection,
        "mlp_macs_per_block": mlp_first + mlp_second,
        "transformer_macs_total": transformer_total,
        "classifier_macs": classifier,
        "total_macs": total_macs,
        "total_flops": total_flops,
    }

    return total_macs, total_flops, details


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def copy_state_to_cpu(model):
    """Snapshot current model weights without changing the model class."""

    return {
        name: cp.asnumpy(value).copy()
        for name, value in model.state_dict().items()
    }


def restore_state(model, cpu_state):
    """Copy a saved state back into the arrays already owned by the model."""

    current_state = model.state_dict()

    for name, target in current_state.items():
        if name not in cpu_state:
            raise KeyError(f"Missing parameter in saved state: {name}")

        source = cpu_state[name]

        if tuple(target.shape) != tuple(source.shape):
            raise ValueError(f"Shape mismatch for {name}: model={target.shape}, " f"checkpoint={source.shape}")

        target[...] = cp.asarray(source, dtype=target.dtype)


def save_checkpoint(model, config, class_names, mean, std, best_epoch, best_val_accuracy):
    checkpoint_path = CHECKPOINT_DIR / f"{config['name']}.npz"

    state = {
        "experiment_name": np.asarray(config["name"]),
        "num_blocks": np.asarray(config["num_blocks"], dtype=np.int32),
        "num_heads": np.asarray(config["num_heads"], dtype=np.int32),
        "embed_dim": np.asarray(config["embed_dim"], dtype=np.int32),
        "mlp_ratio": np.asarray(config["mlp_ratio"], dtype=np.int32),
        "best_epoch": np.asarray(best_epoch, dtype=np.int32),
        "best_val_accuracy": np.asarray(best_val_accuracy, dtype=np.float32),
        "class_names": np.asarray(class_names),
        "rgb_mean": np.asarray(mean, dtype=np.float32),
        "rgb_std": np.asarray(std, dtype=np.float32),
    }

    for name, value in model.state_dict().items():
        state[name] = cp.asnumpy(value)

    np.savez(checkpoint_path, **state)

    return checkpoint_path


# ============================================================================
# HISTORY / CSV
# ============================================================================

def save_history_csv(config, history):
    path = HISTORY_DIR / f"{config['name']}.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "train_objective",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_seconds",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return path


def save_summary_csv(results):
    path = RESULTS_DIR / "results_summary.csv"

    if not results:
        return path

    fieldnames = list(results[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return path


# ============================================================================
# PLOTS
# ============================================================================

def plot_history(config, history):
    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epoch - {config['name']}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{config['name']}_accuracy_history.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train CE Loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch - {config['name']}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{config['name']}_loss_history.png", dpi=180)
    plt.close()


def plot_xy(results, x_key, y_key, title, xlabel, ylabel, filename):
    if len(results) < 2:
        return

    x = [row[x_key] for row in results]
    y = [row[y_key] for row in results]
    labels = [row["name"] for row in results]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=60)
    plt.plot(x, y, alpha=0.5)

    for xv, yv, label in zip(x, y, labels):
        plt.annotate(label, (xv, yv), xytext=(5, 5), textcoords="offset points", fontsize=8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=180)
    plt.close()


def results_for_sweep(all_results, sweep):
    """Return baseline + rows belonging to one controlled sweep."""

    rows = [row for row in all_results if row["sweep"] == sweep]
    baseline_rows = [row for row in all_results if row["name"] == "baseline"]

    if baseline_rows:
        rows = baseline_rows + rows

    if sweep == "layers":
        return sorted(rows, key=lambda r: r["num_blocks"])
    if sweep == "heads":
        return sorted(rows, key=lambda r: r["num_heads"])
    if sweep == "embed_dim":
        return sorted(rows, key=lambda r: r["embed_dim"])
    if sweep == "mlp_ratio":
        return sorted(rows, key=lambda r: r["mlp_ratio"])

    return rows


def make_all_plots(results):
    # Overall accuracy/computation trade-offs.
    plot_xy(
        results,
        "parameters_million",
        "best_val_accuracy",
        "Validation Accuracy vs Number of Parameters",
        "Parameters (millions)",
        "Best Validation Accuracy",
        "accuracy_vs_parameters.png",
    )

    plot_xy(results, "gflops", "best_val_accuracy", "Validation Accuracy vs Compute", "Forward GFLOPs", "Best Validation Accuracy", "accuracy_vs_gflops.png")

    plot_xy(
        results,
        "latency_ms_batch1",
        "best_val_accuracy",
        "Validation Accuracy vs Inference Latency",
        "Batch-1 Latency (ms/image)",
        "Best Validation Accuracy",
        "accuracy_vs_latency.png",
    )

    plot_xy(
        results,
        "throughput_images_per_second",
        "best_val_accuracy",
        "Validation Accuracy vs Inference Throughput",
        "Throughput (images/second)",
        "Best Validation Accuracy",
        "accuracy_vs_throughput.png",
    )

    sweep_specs = {
        "layers": ("num_blocks", "Number of Transformer Blocks", "layers"),
        "heads": ("num_heads", "Number of Attention Heads", "heads"),
        "embed_dim": ("embed_dim", "Embedding Dimension", "embedding_dimension"),
        "mlp_ratio": ("mlp_ratio", "MLP Ratio", "mlp_ratio"),
    }

    for sweep, (x_key, x_label, prefix) in sweep_specs.items():
        rows = results_for_sweep(results, sweep)
        if len(rows) < 2:
            continue

        plot_xy(rows, x_key, "best_val_accuracy", f"{x_label} vs Validation Accuracy", x_label, "Best Validation Accuracy", f"{prefix}_vs_accuracy.png")

        plot_xy(rows, x_key, "parameters_million", f"{x_label} vs Model Parameters", x_label, "Parameters (millions)", f"{prefix}_vs_parameters.png")

        plot_xy(rows, x_key, "gflops", f"{x_label} vs Compute", x_label, "Forward GFLOPs", f"{prefix}_vs_gflops.png")

        plot_xy(rows, x_key, "latency_ms_batch1", f"{x_label} vs Inference Latency", x_label, "Batch-1 Latency (ms/image)", f"{prefix}_vs_latency.png")

        plot_xy(rows, x_key, "throughput_images_per_second", f"{x_label} vs Throughput", x_label, "Throughput (images/second)", f"{prefix}_vs_throughput.png")


# ============================================================================
# GPU INFERENCE BENCHMARK
# ============================================================================

def benchmark_inference(model):
    """
    Measure model-only GPU inference.

    Disk I/O and PIL preprocessing are intentionally excluded.
    CUDA events are used because GPU kernels execute asynchronously.
    """

    # ------------------------------------------------------------
    # BATCH-1 LATENCY
    # ------------------------------------------------------------
    x1 = cp.zeros((1, core1.IMAGE_SIZE, core1.IMAGE_SIZE, core1.CHANNELS), dtype=cp.float32)

    for _ in range(LATENCY_WARMUP_RUNS):
        model.forward(x1, training=False)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(LATENCY_MEASURE_RUNS):
        model.forward(x1, training=False)
    end.record()
    end.synchronize()

    total_ms = float(cp.cuda.get_elapsed_time(start, end))
    latency_ms_batch1 = total_ms / LATENCY_MEASURE_RUNS

    # ------------------------------------------------------------
    # BATCHED THROUGHPUT
    # ------------------------------------------------------------
    batch_size = THROUGHPUT_BATCH_SIZE

    xb = cp.zeros((batch_size, core1.IMAGE_SIZE, core1.IMAGE_SIZE, core1.CHANNELS), dtype=cp.float32)

    for _ in range(LATENCY_WARMUP_RUNS):
        model.forward(xb, training=False)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(LATENCY_MEASURE_RUNS):
        model.forward(xb, training=False)
    end.record()
    end.synchronize()

    total_batch_ms = float(cp.cuda.get_elapsed_time(start, end))
    average_batch_ms = total_batch_ms / LATENCY_MEASURE_RUNS

    throughput = batch_size / (average_batch_ms / 1000.0)

    del x1, xb

    return latency_ms_batch1, average_batch_ms, throughput


# ============================================================================
# TRAIN ONE CONTROLLED MODEL
# ============================================================================

def train_one_experiment(config, class_names, train_samples, val_samples, mean, std):
    """Train one architecture using the exact Core Component 1 pipeline."""

    print()
    print("=" * 78)
    print(f"CORE COMPONENT 2 EXPERIMENT: {config['name']}")
    print("=" * 78)
    print(f"Blocks               : {config['num_blocks']}")
    print(f"Attention heads      : {config['num_heads']}")
    print(f"Embedding dimension  : {config['embed_dim']}")
    print(f"Head dimension       : {config['embed_dim'] // config['num_heads']}")
    print(f"MLP ratio            : {config['mlp_ratio']}")
    print(f"MLP hidden dimension : {config['embed_dim'] * config['mlp_ratio']}")
    print(f"Epochs               : {EXPERIMENT_EPOCHS}")
    print("All non-capacity training settings are reused from core_component1.py")
    print("=" * 78)

    reset_random_seeds()

    num_classes = len(class_names)
    num_train = len(train_samples)
    steps_per_epoch = math.ceil(num_train / core1.BATCH_SIZE)
    total_steps = EXPERIMENT_EPOCHS * steps_per_epoch
    warmup_steps = core1.WARMUP_EPOCHS * steps_per_epoch

    model = build_model(num_classes, config)

    # Exact model size.
    parameter_count, parameter_breakdown = count_parameters(model)

    # Theoretical forward compute.
    macs, flops, compute_details = calculate_macs_flops(num_classes, config)

    print(f"Parameters           : {parameter_count:,}")
    print(f"Parameters (M)       : {parameter_count / 1e6:.4f}")
    print(f"Forward MACs         : {macs:,}")
    print(f"Forward GMACs        : {macs / 1e9:.6f}")
    print(f"Forward FLOPs        : {flops:,}")
    print(f"Forward GFLOPs       : {flops / 1e9:.6f}")
    print()

    # Reset data-order RNGs identically for each architecture.
    shuffle_rng = random.Random(core1.SEED)
    augmentation_rng = random.Random(core1.SEED + 100000)

    # Reset CuPy RNG after parameter initialization so training-time stochastic
    # operations start from a reproducible seed for every model.
    cp.random.seed(core1.SEED + 200000)

    history = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    global_step = 0

    experiment_start = time.perf_counter()

    save_progress(
        "experiment_started",
        experiment=config["name"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        embed_dim=config["embed_dim"],
        mlp_ratio=config["mlp_ratio"],
        epochs_total=EXPERIMENT_EPOCHS,
    )

    for epoch in range(1, EXPERIMENT_EPOCHS + 1):
        epoch_start = time.perf_counter()

        epoch_loss_sum = 0.0
        epoch_objective_sum = 0.0
        epoch_correct = 0
        epoch_seen = 0

        batch_iterator = core1.create_epoch_batches(train_samples, core1.BATCH_SIZE, shuffle_rng)

        for step, batch_samples in enumerate(batch_iterator, start=1):
            images, labels = core1.load_batch_to_gpu(batch_samples, mean, std, training=True, rng=augmentation_rng, epoch=epoch)

            logits = model.forward(images, training=True)

            objective_loss, hard_ce_loss, dlogits, predictions = core1.cross_entropy_forward_backward(logits, labels, label_smoothing=core1.LABEL_SMOOTHING)

            model.backward(dlogits)

            current_lr = core1.get_learning_rate(global_step, total_steps, warmup_steps)

            model.adamw_step(
                learning_rate=current_lr,
                beta1=core1.ADAM_BETA1,
                beta2=core1.ADAM_BETA2,
                adam_eps=core1.ADAM_EPS,
                weight_decay=core1.WEIGHT_DECAY,
            )

            global_step += 1

            B = int(labels.shape[0])
            batch_correct = int(cp.sum(predictions == labels).get())

            epoch_loss_sum += float(hard_ce_loss.get()) * B
            epoch_objective_sum += float(objective_loss.get()) * B
            epoch_correct += batch_correct
            epoch_seen += B

            if step == 1 or step % 10 == 0 or step == steps_per_epoch:
                print(
                    f"{config['name']} | "
                    f"Epoch {epoch:02d}/{EXPERIMENT_EPOCHS:02d} | "
                    f"Step {step:04d}/{steps_per_epoch:04d} | "
                    f"LR {current_lr:.6e} | "
                    f"CE {float(hard_ce_loss.get()):.4f} | "
                    f"Smooth {float(objective_loss.get()):.4f} | "
                    f"Batch Acc {batch_correct / B:.4f}"
                )

        train_loss = epoch_loss_sum / epoch_seen
        train_objective = epoch_objective_sum / epoch_seen
        train_accuracy = epoch_correct / epoch_seen

        # Reuse Core Component 1 validation exactly.
        val_loss, val_accuracy = core1.evaluate(model, val_samples, mean, std)

        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_objective": train_objective,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        # Persist every completed epoch immediately. If the terminal/session
        # closes later, all completed epoch metrics still remain on disk.
        history_path = save_history_csv(config, history)
        plot_history(config, history)
        save_progress(
            "training",
            experiment=config["name"],
            epoch=epoch,
            epochs_total=EXPERIMENT_EPOCHS,
            history_csv=str(history_path),
            train_accuracy=float(train_accuracy),
            val_accuracy=float(val_accuracy),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
        )

        print()
        print(f"{config['name']} | Epoch {epoch:02d} summary")
        print(f"Train Loss           : {train_loss:.4f}")
        print(f"Train Objective      : {train_objective:.4f}")
        print(f"Train Accuracy       : {train_accuracy:.4f}")
        print(f"Validation Loss      : {val_loss:.4f}")
        print(f"Validation Accuracy  : {val_accuracy:.4f}")
        print(f"Accuracy Gap         : {train_accuracy - val_accuracy:.4f}")
        print(f"Epoch Time           : {epoch_seconds:.2f} s")

        # Save the best validation-accuracy model for fair inference comparison.
        if not math.isnan(val_accuracy) and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy_state_to_cpu(model)

            # Persist the best weights immediately, not only after the whole
            # experiment finishes. This protects long runs from terminal loss.
            best_checkpoint_path = save_checkpoint(model, config, class_names, mean, std, best_epoch, best_val_accuracy)

            save_progress(
                "training_best_checkpoint_saved",
                experiment=config["name"],
                epoch=epoch,
                best_val_accuracy=float(best_val_accuracy),
                checkpoint=str(best_checkpoint_path),
                history_csv=str(history_path),
            )
            print(f"New best validation checkpoint saved: {best_checkpoint_path}")

        print("-" * 78)

    total_training_seconds = time.perf_counter() - experiment_start

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was produced.")

    # Restore best validation-accuracy state BEFORE latency/throughput benchmark.
    restore_state(model, best_state)

    checkpoint_path = save_checkpoint(model, config, class_names, mean, std, best_epoch, best_val_accuracy)

    history_path = save_history_csv(config, history)
    plot_history(config, history)

    # Benchmark after restoring the best checkpoint.
    latency_ms_batch1, throughput_batch_ms, throughput = benchmark_inference(model)

    final_row = history[-1]
    best_history_row = history[best_epoch - 1]

    result = {
        "name": config["name"],
        "sweep": config["sweep"],
        "sweep_value": config["sweep_value"],
        "num_blocks": config["num_blocks"],
        "num_heads": config["num_heads"],
        "head_dim": config["embed_dim"] // config["num_heads"],
        "embed_dim": config["embed_dim"],
        "mlp_ratio": config["mlp_ratio"],
        "mlp_hidden_dim": config["embed_dim"] * config["mlp_ratio"],
        "parameters": parameter_count,
        "parameters_million": parameter_count / 1e6,
        "macs": macs,
        "gmacs": macs / 1e9,
        "flops": flops,
        "gflops": flops / 1e9,
        "best_epoch": best_epoch,
        "best_train_loss": best_history_row["train_loss"],
        "best_train_accuracy": best_history_row["train_accuracy"],
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_accuracy_gap": best_history_row["train_accuracy"] - best_val_accuracy,
        "final_train_loss": final_row["train_loss"],
        "final_train_accuracy": final_row["train_accuracy"],
        "final_val_loss": final_row["val_loss"],
        "final_val_accuracy": final_row["val_accuracy"],
        "training_seconds": total_training_seconds,
        "latency_ms_batch1": latency_ms_batch1,
        "throughput_batch_size": THROUGHPUT_BATCH_SIZE,
        "throughput_batch_latency_ms": throughput_batch_ms,
        "throughput_images_per_second": throughput,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
    }

    # Save useful model/compute breakdowns for auditability.
    breakdown_path = RESULTS_DIR / f"{config['name']}_breakdown.txt"
    with breakdown_path.open("w", encoding="utf-8") as file:
        file.write("PARAMETER BREAKDOWN\n")
        file.write("=" * 60 + "\n")
        for key, value in parameter_breakdown.items():
            file.write(f"{key:35s} {value:15,d}\n")
        file.write(f"{'TOTAL':35s} {parameter_count:15,d}\n\n")

        file.write("MAC/FLOP BREAKDOWN\n")
        file.write("=" * 60 + "\n")
        for key, value in compute_details.items():
            file.write(f"{key:40s} {value}\n")

    print()
    print(f"Completed {config['name']}")
    print(f"Best Val Accuracy    : {best_val_accuracy:.4f} (epoch {best_epoch})")
    print(f"Latency (batch=1)    : {latency_ms_batch1:.4f} ms")
    print(f"Throughput           : {throughput:.2f} images/s")
    print(f"Training time        : {total_training_seconds / 60.0:.2f} min")
    print(f"Checkpoint           : {checkpoint_path}")
    print()

    # Save one self-contained result file per completed architecture.
    result_json_path = RESULTS_DIR / f"{config['name']}_result.json"
    with result_json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.flush()

    save_progress(
        "experiment_completed",
        experiment=config["name"],
        best_epoch=best_epoch,
        best_val_accuracy=float(best_val_accuracy),
        checkpoint=str(checkpoint_path),
        history_csv=str(history_path),
        result_json=str(result_json_path),
    )

    # Release this architecture before constructing the next one.
    del best_state
    del model
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return result


# ============================================================================
# CONCLUSION / PARETO ANALYSIS
# ============================================================================

def pareto_frontier_accuracy_vs_compute(results):
    """Maximize validation accuracy while minimizing GFLOPs."""

    frontier = []

    for candidate in results:
        dominated = False

        for other in results:
            if other is candidate:
                continue

            no_more_compute = other["gflops"] <= candidate["gflops"]
            no_less_accuracy = other["best_val_accuracy"] >= candidate["best_val_accuracy"]
            strictly_better_somewhere = (other["gflops"] < candidate["gflops"] or other["best_val_accuracy"] > candidate["best_val_accuracy"])

            if no_more_compute and no_less_accuracy and strictly_better_somewhere:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return sorted(frontier, key=lambda row: row["gflops"])


def write_conclusions(results):
    path = RESULTS_DIR / "conclusions.md"

    best_accuracy = max(results, key=lambda r: r["best_val_accuracy"])
    smallest = min(results, key=lambda r: r["parameters"])
    lowest_compute = min(results, key=lambda r: r["gflops"])
    lowest_latency = min(results, key=lambda r: r["latency_ms_batch1"])
    highest_throughput = max(results, key=lambda r: r["throughput_images_per_second"])
    pareto = pareto_frontier_accuracy_vs_compute(results)

    baseline = next(row for row in results if row["name"] == "baseline")

    with path.open("w", encoding="utf-8") as file:
        file.write("# Core Component 2 - Capacity Scaling Results\n\n")
        file.write("All experiments use the same dataset split, preprocessing, augmentation, optimizer, learning-rate schedule, batch size, seed and epoch budget. Only the listed architectural capacity variable is changed within each sweep.\n\n")

        file.write("## Baseline\n\n")
        file.write(
            f"- Blocks: {baseline['num_blocks']}\n"
            f"- Heads: {baseline['num_heads']}\n"
            f"- Embedding dimension: {baseline['embed_dim']}\n"
            f"- MLP ratio: {baseline['mlp_ratio']}\n"
            f"- Parameters: {baseline['parameters']:,}\n"
            f"- GFLOPs: {baseline['gflops']:.6f}\n"
            f"- Best validation accuracy: {baseline['best_val_accuracy']:.4f}\n\n"
        )

        file.write("## Overall extrema\n\n")
        file.write(f"- Best validation accuracy: **{best_accuracy['name']}** = {best_accuracy['best_val_accuracy']:.4f}\n")
        file.write(f"- Fewest parameters: **{smallest['name']}** = {smallest['parameters']:,}\n")
        file.write(f"- Lowest compute: **{lowest_compute['name']}** = {lowest_compute['gflops']:.6f} GFLOPs\n")
        file.write(f"- Lowest batch-1 latency: **{lowest_latency['name']}** = {lowest_latency['latency_ms_batch1']:.4f} ms\n")
        file.write(f"- Highest throughput: **{highest_throughput['name']}** = {highest_throughput['throughput_images_per_second']:.2f} images/s\n\n")

        file.write("## Accuracy-compute Pareto frontier\n\n")
        file.write("A model is on this frontier if no other tested model has both equal-or-better accuracy and equal-or-lower GFLOPs.\n\n")
        file.write("| Model | Best Val Acc | GFLOPs | Params (M) | Latency ms | Throughput img/s |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for row in pareto:
            file.write(
                f"| {row['name']} | {row['best_val_accuracy']:.4f} | "
                f"{row['gflops']:.6f} | {row['parameters_million']:.4f} | "
                f"{row['latency_ms_batch1']:.4f} | "
                f"{row['throughput_images_per_second']:.2f} |\n"
            )

        file.write("\n## Controlled sweep changes relative to baseline\n\n")

        for sweep, title in [
            ("layers", "Number of layers"),
            ("heads", "Attention heads"),
            ("embed_dim", "Embedding dimension"),
            ("mlp_ratio", "MLP ratio"),
        ]:
            rows = results_for_sweep(results, sweep)
            if len(rows) < 2:
                continue

            file.write(f"### {title}\n\n")
            file.write("| Model | Val Acc | ΔAcc | GFLOPs | ΔGFLOPs | Params (M) | Latency ms | Throughput |\n")
            file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

            for row in rows:
                file.write(
                    f"| {row['name']} | {row['best_val_accuracy']:.4f} | "
                    f"{row['best_val_accuracy'] - baseline['best_val_accuracy']:+.4f} | "
                    f"{row['gflops']:.6f} | "
                    f"{row['gflops'] - baseline['gflops']:+.6f} | "
                    f"{row['parameters_million']:.4f} | "
                    f"{row['latency_ms_batch1']:.4f} | "
                    f"{row['throughput_images_per_second']:.2f} |\n"
                )

            file.write("\n")

        file.write("## Interpretation note for attention-head sweep\n\n")
        file.write(
            "At fixed embedding dimension D, changing only the number of heads changes head_dim = D / H. "
            "The Q/K/V/O projection matrices remain D x D, and H*T*T*(D/H) = T*T*D, so the theoretical parameter count and dominant attention MAC count remain essentially unchanged. "
            "Measured latency/throughput can still change because the GPU sees different tensor shapes.\n"
        )

    return path


# ============================================================================
# PRETTY FINAL TABLE
# ============================================================================

def print_final_table(results):
    print()
    print("=" * 150)
    print("CORE COMPONENT 2 - FINAL CAPACITY COMPARISON")
    print("=" * 150)

    header = (
        f"{'Model':<18} "
        f"{'L':>3} {'H':>3} {'D':>5} {'MLP':>4} "
        f"{'Params(M)':>10} {'GMACs':>10} {'GFLOPs':>10} "
        f"{'TrainAcc':>10} {'ValAcc':>10} "
        f"{'Latency(ms)':>12} {'Imgs/s':>12}"
    )
    print(header)
    print("-" * 150)

    for row in results:
        print(
            f"{row['name']:<18} "
            f"{row['num_blocks']:>3d} "
            f"{row['num_heads']:>3d} "
            f"{row['embed_dim']:>5d} "
            f"{row['mlp_ratio']:>4d} "
            f"{row['parameters_million']:>10.4f} "
            f"{row['gmacs']:>10.6f} "
            f"{row['gflops']:>10.6f} "
            f"{row['best_train_accuracy']:>10.4f} "
            f"{row['best_val_accuracy']:>10.4f} "
            f"{row['latency_ms_batch1']:>12.4f} "
            f"{row['throughput_images_per_second']:>12.2f}"
        )

    print("=" * 150)


# ============================================================================
# MAIN CORE COMPONENT 2 EXPERIMENT
# ============================================================================

def run_core_component2():
    print("=" * 78)
    print("CORE COMPONENT 2: CONTROLLED MODEL CAPACITY EXPERIMENTS")
    print("Model implementation in model/ is NOT modified.")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # 1. EXACT SAME DATASET SPLIT AS CORE COMPONENT 1
    # ------------------------------------------------------------------------
    class_names, train_samples, val_samples, train_class_counts = core1.build_dataset(core1.TRAIN_PATH, val_fraction=core1.VAL_FRACTION, seed=core1.SEED)

    # ------------------------------------------------------------------------
    # 2. EXACT SAME TRAINING-ONLY NORMALIZATION STATISTICS
    # ------------------------------------------------------------------------
    mean, std = core1.get_training_mean_std(train_samples)

    print(f"Classes               : {len(class_names)}")
    print(f"Training images       : {len(train_samples)}")
    print(f"Validation images     : {len(val_samples)}")
    print(f"Batch size            : {core1.BATCH_SIZE}")
    print(f"Epoch budget/model    : {EXPERIMENT_EPOCHS}")
    print(f"Optimizer             : AdamW")
    print(f"Max/Min LR            : {core1.MAX_LEARNING_RATE} / {core1.MIN_LEARNING_RATE}")
    print(f"Warmup epochs         : {core1.WARMUP_EPOCHS}")
    print(f"Weight decay          : {core1.WEIGHT_DECAY}")
    print(f"Dropout               : {core1.DROPOUT_RATE}")
    print(f"Embedding dropout     : {core1.EMBEDDING_DROPOUT_RATE}")
    print(f"Label smoothing       : {core1.LABEL_SMOOTHING}")
    print(f"Seed                  : {core1.SEED}")
    print()

    experiments = make_experiment_list()

    print("Experiments:")
    for index, cfg in enumerate(experiments, start=1):
        print(f"  {index:02d}. {cfg['name']:<18} | " f"L={cfg['num_blocks']} H={cfg['num_heads']} " f"D={cfg['embed_dim']} MLP={cfg['mlp_ratio']}")

    results = []

    for config in experiments:
        result = train_one_experiment(config, class_names, train_samples, val_samples, mean, std)

        results.append(result)

        # Save after every completed model so long experiment runs are not lost.
        summary_path = save_summary_csv(results)
        make_all_plots(results)
        save_progress(
            "model_completed",
            experiment=config["name"],
            completed_models=len(results),
            total_models=len(experiments),
            results_csv=str(summary_path),
        )

    summary_path = save_summary_csv(results)
    make_all_plots(results)
    conclusion_path = write_conclusions(results)

    print_final_table(results)

    print()
    print("Core Component 2 complete.")
    print(f"Results CSV : {summary_path}")
    print(f"Plots       : {PLOTS_DIR}")
    print(f"Histories   : {HISTORY_DIR}")
    print(f"Checkpoints : {CHECKPOINT_DIR}")
    print(f"Conclusions : {conclusion_path}")

    save_progress(
        "completed",
        completed_models=len(results),
        total_models=len(experiments),
        results_csv=str(summary_path),
        plots_dir=str(PLOTS_DIR),
        history_dir=str(HISTORY_DIR),
        checkpoint_dir=str(CHECKPOINT_DIR),
        conclusions=str(conclusion_path),
    )


if __name__ == "__main__":
    log_path, log_file, original_stdout, original_stderr = start_persistent_logging()

    try:
        print(f"Persistent run log   : {log_path}")
        save_progress("run_started", log_file=str(log_path))
        run_core_component2()
    except BaseException as error:
        # Save the traceback and failure state before propagating the error.
        print()
        print("CORE COMPONENT 2 TERMINATED WITH AN ERROR")
        traceback.print_exc()
        save_progress("failed", error_type=type(error).__name__, error_message=str(error), log_file=str(log_path))
        raise
    finally:
        stop_persistent_logging(log_file, original_stdout, original_stderr)
