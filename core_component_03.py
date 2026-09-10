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

# Existing training/data pipeline. Importing it does not call train(), because
# its train() invocation is protected by if __name__ == '__main__'.
# Support either naming convention used in the project.
try:
    import core_component_01 as core1
except ModuleNotFoundError:
    import core_component_01 as core1

# Exact same model implementation. Nothing inside model/ is changed.
from model.vit_architecture import VisionTransformer


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "core_component3_results"
HISTORY_DIR = RESULTS_DIR / "history"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"
SAMPLING_DIR = RESULTS_DIR / "sampling"

for directory in (
    RESULTS_DIR,
    HISTORY_DIR,
    PLOTS_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    SAMPLING_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# Same architecture for EVERY experiment.
BASELINE_ARCHITECTURE = {
    "image_size": core1.IMAGE_SIZE,
    "patch_size": core1.PATCH_SIZE,
    "in_channels": core1.CHANNELS,
    "embed_dim": core1.EMBED_DIM,
    "num_heads": core1.NUM_HEADS,
    "num_blocks": core1.NUM_BLOCKS,
    "mlp_ratio": core1.MLP_RATIO,
    "eps": 1e-5,
    "dropout_rate": core1.DROPOUT_RATE,
    "embedding_dropout_rate": core1.EMBEDDING_DROPOUT_RATE,
}

# Main data-scaling experiment requested in the assignment.
DATA_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
SAMPLING_STRATEGIES = ["uniform", "nonuniform"]

# Nonuniform class preference strength.
# 1.0 would be uniform. 4.0 means the most-favored class receives a sampling
# weight four times the least-favored class before weighted sampling.
NONUNIFORM_MAX_TO_MIN_WEIGHT_RATIO = 4.0

# A fixed training budget gives the cleanest data-size comparison. Different
# runs do not stop after different numbers of epochs simply because one subset
# happened to plateau earlier.
EXPERIMENT_EPOCHS = core1.EPOCHS
USE_EARLY_STOPPING = False

# Set True to skip experiments that already have a saved result JSON.
# Useful if a long terminal/server job is interrupted.
RESUME_COMPLETED_EXPERIMENTS = True

# Plot image quality.
PLOT_DPI = 160


# =============================================================================
# PERSISTENT LOGGING
# =============================================================================

class TeeStream:
    """Write terminal output to both terminal and a persistent file."""

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{timestamp}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    return log_path, log_file, original_stdout, original_stderr


def stop_persistent_logging(log_file, original_stdout, original_stderr):
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()


def save_progress(status, **extra):
    progress = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    progress.update(extra)

    path = RESULTS_DIR / "progress.json"
    temporary = RESULTS_DIR / "progress.json.tmp"

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(progress, file, indent=2)
        file.flush()

    temporary.replace(path)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def reset_random_seeds():
    random.seed(core1.SEED)
    np.random.seed(core1.SEED)
    cp.random.seed(core1.SEED)


def synchronize_gpu():
    cp.cuda.Stream.null.synchronize()


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def fraction_tag(fraction):
    return f"{int(round(100.0 * fraction)):03d}pct"


def experiment_name(strategy, fraction):
    return f"{strategy}_{fraction_tag(fraction)}"


def canonicalize_samples(samples):
    """Use a stable order before the epoch-level shuffle."""
    return sorted(samples, key=lambda item: (item[1], str(item[0])))


def group_samples_by_class(samples, num_classes):
    groups = [[] for _ in range(num_classes)]
    for sample in samples:
        groups[int(sample[1])].append(sample)
    return groups


def subset_class_counts(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in samples:
        counts[int(label)] += 1
    return counts


def subset_distribution_metrics(counts):
    positive = counts[counts > 0]

    represented = int(positive.size)

    if positive.size == 0:
        imbalance_ratio = float("nan")
    else:
        imbalance_ratio = float(positive.max() / positive.min())

    return represented, imbalance_ratio


# =============================================================================
# NESTED UNIFORM / STRATIFIED SAMPLING
# =============================================================================

def prepare_uniform_class_orders(train_samples, num_classes):
    """
    Shuffle each class independently once.

    Later fractions take prefixes of these fixed class orders. Therefore a
    smaller subset is nested inside a larger subset.
    """

    groups = group_samples_by_class(train_samples, num_classes)
    ordered_groups = []

    for class_idx, group in enumerate(groups):
        class_group = list(group)
        rng = random.Random(core1.SEED + 1000 + class_idx)
        rng.shuffle(class_group)
        ordered_groups.append(class_group)

    return ordered_groups


def uniform_stratified_subset(ordered_groups, fraction):
    """
    Keep approximately the same fraction from every class.

    The total subset size is made as close as possible to round(fraction*N)
    while preserving stratification.
    """

    class_sizes = np.asarray([len(group) for group in ordered_groups], dtype=np.int64)
    full_size = int(class_sizes.sum())
    target_total = int(round(full_size * fraction))

    if target_total <= 0:
        raise ValueError("Data fraction produced an empty training subset.")

    desired = class_sizes.astype(np.float64) * float(fraction)
    allocation = np.floor(desired).astype(np.int64)

    # If the requested total can include each non-empty class, guarantee at
    # least one example from every class for the uniform strategy.
    nonempty_classes = np.where(class_sizes > 0)[0]
    if target_total >= len(nonempty_classes):
        for class_idx in nonempty_classes:
            if allocation[class_idx] == 0:
                allocation[class_idx] = 1

    allocation = np.minimum(allocation, class_sizes)

    current = int(allocation.sum())

    # Add samples until exact target is reached. Prioritize classes with the
    # largest fractional remainder from n_class * fraction.
    if current < target_total:
        remainder = desired - np.floor(desired)
        order = np.argsort(-remainder)

        while current < target_total:
            changed = False
            for class_idx in order:
                if allocation[class_idx] < class_sizes[class_idx]:
                    allocation[class_idx] += 1
                    current += 1
                    changed = True
                    if current >= target_total:
                        break
            if not changed:
                break

    # If minimum-one logic overshot the exact target, remove from classes with
    # the smallest fractional remainder while never going below zero.
    elif current > target_total:
        remainder = desired - np.floor(desired)
        order = np.argsort(remainder)

        while current > target_total:
            changed = False
            for class_idx in order:
                minimum = 1 if target_total >= len(nonempty_classes) and class_sizes[class_idx] > 0 else 0
                if allocation[class_idx] > minimum:
                    allocation[class_idx] -= 1
                    current -= 1
                    changed = True
                    if current <= target_total:
                        break
            if not changed:
                break

    subset = []
    for class_idx, group in enumerate(ordered_groups):
        subset.extend(group[: int(allocation[class_idx])])

    return canonicalize_samples(subset)


# =============================================================================
# NESTED NON-UNIFORM SAMPLING
# =============================================================================

def prepare_nonuniform_weighted_order(train_samples, num_classes):
    """
    Build one weighted random permutation of all training samples.

    Classes receive fixed weights between 1 and
    NONUNIFORM_MAX_TO_MIN_WEIGHT_RATIO. Which class gets which weight is itself
    shuffled with the project seed so class-folder alphabetical order does not
    create the imbalance.

    A weighted random priority is generated as:
        priority = -log(U) / class_weight

    Lower priority is selected first. Taking prefixes of the resulting order
    gives nested nonuniform subsets without replacement.
    """

    rng = np.random.RandomState(core1.SEED + 200000)

    class_order = np.arange(num_classes, dtype=np.int64)
    rng.shuffle(class_order)

    # Geometric spacing gives a smooth ratio from most- to least-favored class.
    ranked_weights = np.geomspace(
        NONUNIFORM_MAX_TO_MIN_WEIGHT_RATIO,
        1.0,
        num=max(1, num_classes),
    )

    class_weights = np.ones(num_classes, dtype=np.float64)
    for rank, class_idx in enumerate(class_order):
        class_weights[class_idx] = ranked_weights[rank]

    priorities = []

    for sample_index, sample in enumerate(train_samples):
        label = int(sample[1])
        weight = float(class_weights[label])

        # Avoid log(0).
        u = max(float(rng.random_sample()), 1e-12)
        priority = -math.log(u) / weight

        priorities.append((priority, sample_index))

    priorities.sort(key=lambda item: item[0])
    ordered_samples = [train_samples[index] for _, index in priorities]

    return ordered_samples, class_weights


def nonuniform_subset(weighted_order, fraction):
    target_total = int(round(len(weighted_order) * fraction))
    target_total = max(1, min(len(weighted_order), target_total))
    return canonicalize_samples(weighted_order[:target_total])


# =============================================================================
# SAVE SAMPLING DETAILS
# =============================================================================

def save_class_distribution(experiment, class_names, counts, class_weights=None):
    path = SAMPLING_DIR / f"{experiment}_class_counts.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_index", "class_name", "training_samples", "nonuniform_weight"])

        for class_idx, class_name in enumerate(class_names):
            weight = ""
            if class_weights is not None:
                weight = float(class_weights[class_idx])

            writer.writerow([
                class_idx,
                class_name,
                int(counts[class_idx]),
                weight,
            ])

    return path


# =============================================================================
# MODEL CREATION — ARCHITECTURE NEVER CHANGES
# =============================================================================

def build_baseline_model(num_classes):
    return VisionTransformer(
        num_classes=num_classes,
        image_size=BASELINE_ARCHITECTURE["image_size"],
        patch_size=BASELINE_ARCHITECTURE["patch_size"],
        in_channels=BASELINE_ARCHITECTURE["in_channels"],
        embed_dim=BASELINE_ARCHITECTURE["embed_dim"],
        num_heads=BASELINE_ARCHITECTURE["num_heads"],
        num_blocks=BASELINE_ARCHITECTURE["num_blocks"],
        mlp_ratio=BASELINE_ARCHITECTURE["mlp_ratio"],
        eps=BASELINE_ARCHITECTURE["eps"],
        dropout_rate=BASELINE_ARCHITECTURE["dropout_rate"],
        embedding_dropout_rate=BASELINE_ARCHITECTURE["embedding_dropout_rate"],
    )


# =============================================================================
# VALIDATION WITH MACRO-F1
# =============================================================================

def evaluate_with_macro_f1(model, val_samples, mean, std, num_classes):
    if len(val_samples) == 0:
        return float("nan"), float("nan"), float("nan")

    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    true_count = np.zeros(num_classes, dtype=np.int64)
    predicted_count = np.zeros(num_classes, dtype=np.int64)
    true_positive = np.zeros(num_classes, dtype=np.int64)

    for start in range(0, len(val_samples), core1.BATCH_SIZE):
        batch_samples = val_samples[start:start + core1.BATCH_SIZE]

        images, labels = core1.load_batch_to_gpu(
            batch_samples,
            mean,
            std,
            training=False,
            rng=None,
            epoch=None,
        )

        logits = model.forward(images, training=False)
        batch_size = int(labels.shape[0])

        shifted = logits - cp.max(logits, axis=1, keepdims=True)
        exp_logits = cp.exp(shifted)
        probabilities = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)

        correct_probabilities = probabilities[cp.arange(batch_size), labels]
        batch_loss_sum = cp.sum(
            -cp.log(correct_probabilities + cp.float32(1e-12))
        )

        predictions = cp.argmax(probabilities, axis=1)

        total_loss_sum += float(batch_loss_sum.get())
        total_correct += int(cp.sum(predictions == labels).get())
        total_samples += batch_size

        labels_cpu = cp.asnumpy(labels).astype(np.int64, copy=False)
        predictions_cpu = cp.asnumpy(predictions).astype(np.int64, copy=False)

        true_count += np.bincount(labels_cpu, minlength=num_classes)
        predicted_count += np.bincount(predictions_cpu, minlength=num_classes)

        matched = labels_cpu == predictions_cpu
        if np.any(matched):
            true_positive += np.bincount(
                labels_cpu[matched],
                minlength=num_classes,
            )

    val_loss = total_loss_sum / total_samples
    val_accuracy = total_correct / total_samples

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_count > 0,
    )

    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros(num_classes, dtype=np.float64),
        where=true_count > 0,
    )

    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=denominator > 0,
    )

    valid_classes = true_count > 0
    macro_f1 = float(np.mean(f1[valid_classes])) if np.any(valid_classes) else float("nan")

    return val_loss, val_accuracy, macro_f1


# =============================================================================
# CHECKPOINTS
# =============================================================================

def save_experiment_checkpoint(
    model,
    experiment,
    class_names,
    epoch,
    val_accuracy,
    val_macro_f1,
    mean,
    std,
    fraction,
    strategy,
):
    checkpoint_path = CHECKPOINT_DIR / f"{experiment}.npz"

    state = {
        "experiment": np.asarray(experiment),
        "epoch": np.asarray(epoch, dtype=np.int32),
        "val_accuracy": np.asarray(val_accuracy, dtype=np.float32),
        "val_macro_f1": np.asarray(val_macro_f1, dtype=np.float32),
        "class_names": np.asarray(class_names),
        "rgb_mean": np.asarray(mean, dtype=np.float32),
        "rgb_std": np.asarray(std, dtype=np.float32),
        "data_fraction": np.asarray(fraction, dtype=np.float32),
        "sampling_strategy": np.asarray(strategy),
        "image_size": np.asarray(core1.IMAGE_SIZE, dtype=np.int32),
        "patch_size": np.asarray(core1.PATCH_SIZE, dtype=np.int32),
        "embed_dim": np.asarray(core1.EMBED_DIM, dtype=np.int32),
        "num_heads": np.asarray(core1.NUM_HEADS, dtype=np.int32),
        "num_blocks": np.asarray(core1.NUM_BLOCKS, dtype=np.int32),
        "mlp_ratio": np.asarray(core1.MLP_RATIO, dtype=np.int32),
    }

    for name, value in model.state_dict().items():
        state[name] = cp.asnumpy(value)

    np.savez(checkpoint_path, **state)
    return checkpoint_path


# =============================================================================
# HISTORY / RESULT PERSISTENCE
# =============================================================================

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_objective",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "epoch_training_time_seconds",
    "epoch_validation_time_seconds",
]

RESULT_FIELDS = [
    "experiment",
    "strategy",
    "data_fraction",
    "data_percent",
    "num_training_samples",
    "full_training_samples",
    "classes_represented",
    "num_classes",
    "class_coverage",
    "imbalance_ratio",
    "normalization_time_seconds",
    "training_time_seconds",
    "validation_time_seconds",
    "total_wall_time_seconds",
    "epochs_completed",
    "best_epoch",
    "best_val_accuracy",
    "best_val_macro_f1",
    "train_loss_at_best",
    "train_accuracy_at_best",
    "final_train_loss",
    "final_train_objective",
    "final_train_accuracy",
    "final_val_loss",
    "final_val_accuracy",
    "final_val_macro_f1",
    "image_size",
    "patch_size",
    "embed_dim",
    "num_heads",
    "num_blocks",
    "mlp_ratio",
]


def save_history(experiment, history):
    path = HISTORY_DIR / f"{experiment}.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    return path


def result_json_path(experiment):
    return RESULTS_DIR / f"{experiment}_result.json"


def save_result_json(result):
    path = result_json_path(result["experiment"])
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return path


def load_completed_results():
    results = []

    for strategy in SAMPLING_STRATEGIES:
        for fraction in DATA_FRACTIONS:
            name = experiment_name(strategy, fraction)
            path = result_json_path(name)

            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as file:
                        results.append(json.load(file))
                except Exception:
                    pass

    return results


def save_results_summary(results):
    path = RESULTS_DIR / "results_summary.csv"

    ordered = sorted(
        results,
        key=lambda result: (
            result["strategy"],
            float(result["data_fraction"]),
        ),
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        for result in ordered:
            row = {field: result.get(field, "") for field in RESULT_FIELDS}
            writer.writerow(row)

    return path


# =============================================================================
# ONE DATA-SCALING TRAINING RUN
# =============================================================================

def train_one_experiment(
    experiment,
    strategy,
    fraction,
    train_subset,
    val_samples,
    class_names,
    full_training_count,
    class_counts,
):
    num_classes = len(class_names)

    reset_random_seeds()

    # Architecture is reconstructed from exactly the same baseline constants
    # for every subset.
    model = build_baseline_model(num_classes)

    represented_classes, imbalance_ratio = subset_distribution_metrics(class_counts)

    print()
    print("=" * 72)
    print(f"CORE COMPONENT 3 | {experiment}")
    print("=" * 72)
    print(f"Sampling strategy      : {strategy}")
    print(f"Data fraction          : {fraction:.2f} ({100.0 * fraction:.0f}%)")
    print(f"Training samples       : {len(train_subset)} / {full_training_count}")
    print(f"Classes represented    : {represented_classes} / {num_classes}")
    print(f"Class coverage         : {represented_classes / num_classes:.4f}")
    print(f"Imbalance ratio        : {imbalance_ratio:.4f}")
    print(f"Fixed blocks           : {core1.NUM_BLOCKS}")
    print(f"Fixed heads            : {core1.NUM_HEADS}")
    print(f"Fixed embedding dim    : {core1.EMBED_DIM}")
    print(f"Fixed MLP ratio        : {core1.MLP_RATIO}")
    print(f"Epochs                 : {EXPERIMENT_EPOCHS}")
    print("=" * 72)

    save_progress(
        "normalization",
        experiment=experiment,
        strategy=strategy,
        data_fraction=fraction,
        training_samples=len(train_subset),
    )

    # IMPORTANT: statistics are computed ONLY from the subset used by this run,
    # so a 10% experiment does not use pixel statistics from the other 90%.
    normalization_start = time.perf_counter()
    mean, std = core1.compute_training_mean_std(train_subset)
    normalization_time = time.perf_counter() - normalization_start

    print(f"Subset RGB mean         : {mean}")
    print(f"Subset RGB std          : {std}")
    print(f"Normalization time      : {normalization_time:.2f} s")

    steps_per_epoch = math.ceil(len(train_subset) / core1.BATCH_SIZE)
    total_steps = EXPERIMENT_EPOCHS * steps_per_epoch
    warmup_steps = core1.WARMUP_EPOCHS * steps_per_epoch

    shuffle_rng = random.Random(core1.SEED)
    augmentation_rng = random.Random(core1.SEED + 100000)

    history = []
    best_val_accuracy = -1.0
    best_val_macro_f1 = float("nan")
    best_epoch = 0
    best_train_loss = float("nan")
    best_train_accuracy = float("nan")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    cumulative_training_time = 0.0
    cumulative_validation_time = 0.0

    overall_start = time.perf_counter()

    for epoch in range(1, EXPERIMENT_EPOCHS + 1):
        save_progress(
            "training",
            experiment=experiment,
            strategy=strategy,
            data_fraction=fraction,
            epoch=epoch,
            total_epochs=EXPERIMENT_EPOCHS,
        )

        epoch_loss_sum = 0.0
        epoch_objective_sum = 0.0
        epoch_correct = 0
        epoch_seen = 0

        synchronize_gpu()
        epoch_train_start = time.perf_counter()

        batch_iterator = core1.create_epoch_batches(
            train_subset,
            core1.BATCH_SIZE,
            shuffle_rng,
        )

        for step, batch_samples in enumerate(batch_iterator, start=1):
            images, labels = core1.load_batch_to_gpu(
                batch_samples,
                mean,
                std,
                training=True,
                rng=augmentation_rng,
                epoch=epoch,
            )

            logits = model.forward(images, training=True)

            loss, hard_ce_loss, dlogits, predictions = (
                core1.cross_entropy_forward_backward(
                    logits,
                    labels,
                    label_smoothing=core1.LABEL_SMOOTHING,
                )
            )

            model.backward(dlogits)

            current_lr = core1.get_learning_rate(
                global_step,
                total_steps,
                warmup_steps,
            )

            model.adamw_step(
                learning_rate=current_lr,
                beta1=core1.ADAM_BETA1,
                beta2=core1.ADAM_BETA2,
                adam_eps=core1.ADAM_EPS,
                weight_decay=core1.WEIGHT_DECAY,
            )

            global_step += 1

            batch_size = int(labels.shape[0])
            batch_correct = int(cp.sum(predictions == labels).get())

            epoch_loss_sum += float(hard_ce_loss.get()) * batch_size
            epoch_objective_sum += float(loss.get()) * batch_size
            epoch_correct += batch_correct
            epoch_seen += batch_size

            if step == 1 or step % 10 == 0 or step == steps_per_epoch:
                print(
                    f"{experiment} | "
                    f"Epoch {epoch:02d}/{EXPERIMENT_EPOCHS:02d} | "
                    f"Step {step:04d}/{steps_per_epoch:04d} | "
                    f"LR {current_lr:.6e} | "
                    f"CE {float(hard_ce_loss.get()):.4f} | "
                    f"Smooth {float(loss.get()):.4f} | "
                    f"Batch Acc {batch_correct / batch_size:.4f}"
                )

        synchronize_gpu()
        epoch_training_time = time.perf_counter() - epoch_train_start
        cumulative_training_time += epoch_training_time

        train_loss = epoch_loss_sum / epoch_seen
        train_objective = epoch_objective_sum / epoch_seen
        train_accuracy = epoch_correct / epoch_seen

        synchronize_gpu()
        validation_start = time.perf_counter()

        val_loss, val_accuracy, val_macro_f1 = evaluate_with_macro_f1(
            model,
            val_samples,
            mean,
            std,
            num_classes,
        )

        synchronize_gpu()
        epoch_validation_time = time.perf_counter() - validation_start
        cumulative_validation_time += epoch_validation_time

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_objective": float(train_objective),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "val_macro_f1": float(val_macro_f1),
            "epoch_training_time_seconds": float(epoch_training_time),
            "epoch_validation_time_seconds": float(epoch_validation_time),
        }

        history.append(row)
        save_history(experiment, history)

        print()
        print(f"{experiment} | Epoch {epoch:02d} summary")
        print(f"Train Loss            : {train_loss:.4f}")
        print(f"Train Objective       : {train_objective:.4f}")
        print(f"Train Accuracy        : {train_accuracy:.4f}")
        print(f"Validation Loss       : {val_loss:.4f}")
        print(f"Validation Accuracy   : {val_accuracy:.4f}")
        print(f"Validation Macro-F1   : {val_macro_f1:.4f}")
        print(f"Epoch training time   : {epoch_training_time:.2f} s")
        print(f"Epoch validation time : {epoch_validation_time:.2f} s")

        if not math.isnan(val_accuracy) and val_accuracy > best_val_accuracy:
            best_val_accuracy = float(val_accuracy)
            best_val_macro_f1 = float(val_macro_f1)
            best_epoch = epoch
            best_train_loss = float(train_loss)
            best_train_accuracy = float(train_accuracy)

            checkpoint_path = save_experiment_checkpoint(
                model=model,
                experiment=experiment,
                class_names=class_names,
                epoch=epoch,
                val_accuracy=val_accuracy,
                val_macro_f1=val_macro_f1,
                mean=mean,
                std=std,
                fraction=fraction,
                strategy=strategy,
            )

            print(f"Saved new best checkpoint: {checkpoint_path}")

        # Optional only. Disabled by default for controlled scaling.
        if USE_EARLY_STOPPING:
            if (
                not math.isnan(val_loss)
                and val_loss < best_val_loss - core1.EARLY_STOPPING_MIN_DELTA
            ):
                best_val_loss = float(val_loss)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= core1.EARLY_STOPPING_PATIENCE:
                    print("Early stopping: validation loss stopped improving.")
                    break

        save_progress(
            "epoch_completed",
            experiment=experiment,
            strategy=strategy,
            data_fraction=fraction,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_accuracy=best_val_accuracy,
        )

        print("-" * 72)

    synchronize_gpu()
    total_wall_time = time.perf_counter() - overall_start

    final = history[-1]

    result = {
        "experiment": experiment,
        "strategy": strategy,
        "data_fraction": float(fraction),
        "data_percent": float(100.0 * fraction),
        "num_training_samples": int(len(train_subset)),
        "full_training_samples": int(full_training_count),
        "classes_represented": int(represented_classes),
        "num_classes": int(num_classes),
        "class_coverage": float(represented_classes / num_classes),
        "imbalance_ratio": safe_float(imbalance_ratio),
        "normalization_time_seconds": float(normalization_time),
        "training_time_seconds": float(cumulative_training_time),
        "validation_time_seconds": float(cumulative_validation_time),
        "total_wall_time_seconds": float(total_wall_time),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": safe_float(best_val_accuracy),
        "best_val_macro_f1": safe_float(best_val_macro_f1),
        "train_loss_at_best": safe_float(best_train_loss),
        "train_accuracy_at_best": safe_float(best_train_accuracy),
        "final_train_loss": safe_float(final["train_loss"]),
        "final_train_objective": safe_float(final["train_objective"]),
        "final_train_accuracy": safe_float(final["train_accuracy"]),
        "final_val_loss": safe_float(final["val_loss"]),
        "final_val_accuracy": safe_float(final["val_accuracy"]),
        "final_val_macro_f1": safe_float(final["val_macro_f1"]),
        "image_size": int(core1.IMAGE_SIZE),
        "patch_size": int(core1.PATCH_SIZE),
        "embed_dim": int(core1.EMBED_DIM),
        "num_heads": int(core1.NUM_HEADS),
        "num_blocks": int(core1.NUM_BLOCKS),
        "mlp_ratio": int(core1.MLP_RATIO),
    }

    save_result_json(result)

    print()
    print(f"Completed {experiment}")
    print(f"Best epoch             : {best_epoch}")
    print(f"Best validation acc    : {best_val_accuracy:.4f}")
    print(f"Best validation F1     : {best_val_macro_f1:.4f}")
    print(f"Training time          : {cumulative_training_time:.2f} s")
    print(f"Total wall time        : {total_wall_time:.2f} s")

    del model
    cp.get_default_memory_pool().free_all_blocks()

    return result


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def save_line_plot(x_values, series, xlabel, ylabel, title, filename, x_log=False, y_log=False):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for label, y_values in series.items():
        ax.plot(x_values[label], y_values, marker="o", label=label)

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_experiment_history(experiment):
    history_path = HISTORY_DIR / f"{experiment}.csv"
    if not history_path.exists():
        return

    rows = []
    with history_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_acc = [float(row["train_accuracy"]) for row in rows]
    val_acc = [float(row["val_accuracy"]) for row in rows]
    val_f1 = [float(row["val_macro_f1"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, train_acc, marker="o", label="Train accuracy")
    ax.plot(epochs, val_acc, marker="o", label="Validation accuracy")
    ax.plot(epochs, val_f1, marker="o", label="Validation macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Learning curve: {experiment}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"history_scores_{experiment}.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, train_loss, marker="o", label="Train loss")
    ax.plot(epochs, val_loss, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Loss curve: {experiment}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"history_loss_{experiment}.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def results_by_strategy(results):
    grouped = {}
    for strategy in SAMPLING_STRATEGIES:
        group = [result for result in results if result["strategy"] == strategy]
        grouped[strategy] = sorted(group, key=lambda item: float(item["data_fraction"]))
    return grouped


def make_summary_plots(results):
    grouped = results_by_strategy(results)

    def build_series(field):
        x = {}
        y = {}
        for strategy, rows in grouped.items():
            if rows:
                x[strategy] = [float(row["data_percent"]) for row in rows]
                y[strategy] = [float(row[field]) for row in rows]
        return x, y

    plot_specs = [
        (
            "best_val_accuracy",
            "Best validation accuracy",
            "Validation Accuracy vs Training-Data Size",
            "validation_accuracy_vs_data_size.png",
        ),
        (
            "best_val_macro_f1",
            "Best validation macro-F1",
            "Validation Macro-F1 vs Training-Data Size",
            "validation_macro_f1_vs_data_size.png",
        ),
        (
            "final_train_accuracy",
            "Final training accuracy",
            "Training Accuracy vs Training-Data Size",
            "training_accuracy_vs_data_size.png",
        ),
        (
            "final_train_loss",
            "Final training loss",
            "Training Loss vs Training-Data Size",
            "training_loss_vs_data_size.png",
        ),
        (
            "final_val_loss",
            "Final validation loss",
            "Validation Loss vs Training-Data Size",
            "validation_loss_vs_data_size.png",
        ),
        (
            "training_time_seconds",
            "Training time (seconds)",
            "Training Time vs Training-Data Size",
            "training_time_vs_data_size.png",
        ),
        (
            "class_coverage",
            "Class coverage",
            "Class Coverage vs Training-Data Size",
            "class_coverage_vs_data_size.png",
        ),
        (
            "imbalance_ratio",
            "Imbalance ratio (max/min represented class)",
            "Training-Subset Imbalance vs Data Size",
            "imbalance_ratio_vs_data_size.png",
        ),
    ]

    for field, ylabel, title, filename in plot_specs:
        x, y = build_series(field)
        if not y:
            continue
        save_line_plot(
            x_values=x,
            series=y,
            xlabel="Training data used (%)",
            ylabel=ylabel,
            title=title,
            filename=filename,
        )

    # Accuracy vs training time.
    x = {}
    y = {}
    for strategy, rows in grouped.items():
        if rows:
            x[strategy] = [float(row["training_time_seconds"]) for row in rows]
            y[strategy] = [float(row["best_val_accuracy"]) for row in rows]

    if y:
        save_line_plot(
            x_values=x,
            series=y,
            xlabel="Training time (seconds)",
            ylabel="Best validation accuracy",
            title="Accuracy vs Training Time",
            filename="validation_accuracy_vs_training_time.png",
        )


# =============================================================================
# DIMINISHING RETURNS
# =============================================================================

def calculate_diminishing_returns(results):
    grouped = results_by_strategy(results)
    rows_out = []

    for strategy, rows in grouped.items():
        previous = None

        for current in rows:
            if previous is not None:
                additional_samples = (
                    int(current["num_training_samples"])
                    - int(previous["num_training_samples"])
                )

                accuracy_gain = (
                    float(current["best_val_accuracy"])
                    - float(previous["best_val_accuracy"])
                )

                f1_gain = (
                    float(current["best_val_macro_f1"])
                    - float(previous["best_val_macro_f1"])
                )

                gain_per_1000 = (
                    1000.0 * accuracy_gain / additional_samples
                    if additional_samples > 0
                    else float("nan")
                )

                rows_out.append({
                    "strategy": strategy,
                    "from_percent": float(previous["data_percent"]),
                    "to_percent": float(current["data_percent"]),
                    "additional_samples": int(additional_samples),
                    "validation_accuracy_gain": float(accuracy_gain),
                    "validation_macro_f1_gain": float(f1_gain),
                    "accuracy_gain_per_1000_added_samples": float(gain_per_1000),
                })

            previous = current

    path = RESULTS_DIR / "diminishing_returns.csv"
    fields = [
        "strategy",
        "from_percent",
        "to_percent",
        "additional_samples",
        "validation_accuracy_gain",
        "validation_macro_f1_gain",
        "accuracy_gain_per_1000_added_samples",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    # Plot raw accuracy gain for each scaling interval.
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for strategy in SAMPLING_STRATEGIES:
        strategy_rows = [row for row in rows_out if row["strategy"] == strategy]
        if not strategy_rows:
            continue

        labels = [
            f"{int(row['from_percent'])}->{int(row['to_percent'])}%"
            for row in strategy_rows
        ]
        values = [row["validation_accuracy_gain"] for row in strategy_rows]
        ax.plot(labels, values, marker="o", label=strategy)

    ax.set_xlabel("Increase in training-data scale")
    ax.set_ylabel("Gain in best validation accuracy")
    ax.set_title("Diminishing Returns Across Data Scales")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "diminishing_returns.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return rows_out


# =============================================================================
# SIMPLE EMPIRICAL SCALING-LAW FIT
# =============================================================================

def fit_power_law(num_samples, validation_accuracy):
    """
    Fit:
        validation_error = A * N^(-alpha)

    using log-linear least squares:
        log(error) = log(A) - alpha * log(N)

    This is a simple empirical diagnostic. With only four data scales it should
    not be interpreted as proof of a universal scaling law.
    """

    n = np.asarray(num_samples, dtype=np.float64)
    accuracy = np.asarray(validation_accuracy, dtype=np.float64)
    error = 1.0 - accuracy

    valid = (n > 0) & (error > 0) & np.isfinite(error)
    n = n[valid]
    error = error[valid]

    if n.size < 2:
        return None

    x = np.log(n)
    y = np.log(error)

    slope, intercept = np.polyfit(x, y, deg=1)
    alpha = -float(slope)
    A = float(np.exp(intercept))

    predicted_log = intercept + slope * x
    ss_res = float(np.sum((y - predicted_log) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "A": A,
        "alpha": alpha,
        "r_squared_log_space": r_squared,
    }


def analyze_scaling_law(results):
    grouped = results_by_strategy(results)
    fits = []

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for strategy, rows in grouped.items():
        if len(rows) < 2:
            continue

        n = np.asarray(
            [float(row["num_training_samples"]) for row in rows],
            dtype=np.float64,
        )
        accuracy = np.asarray(
            [float(row["best_val_accuracy"]) for row in rows],
            dtype=np.float64,
        )
        error = 1.0 - accuracy

        fit = fit_power_law(n, accuracy)
        if fit is None:
            continue

        fits.append({
            "strategy": strategy,
            "A": fit["A"],
            "alpha": fit["alpha"],
            "r_squared_log_space": fit["r_squared_log_space"],
        })

        ax.plot(n, error, marker="o", linestyle="", label=f"{strategy} measured")

        n_fit = np.geomspace(n.min(), n.max(), 100)
        error_fit = fit["A"] * np.power(n_fit, -fit["alpha"])
        ax.plot(
            n_fit,
            error_fit,
            label=(
                f"{strategy} fit: error ~ N^-{fit['alpha']:.3f}, "
                f"R2={fit['r_squared_log_space']:.3f}"
            ),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of training samples (log scale)")
    ax.set_ylabel("Validation error = 1 - best accuracy (log scale)")
    ax.set_title("Empirical Data Scaling-Law Diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "empirical_scaling_law.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    path = RESULTS_DIR / "scaling_law_fit.csv"
    fields = ["strategy", "A", "alpha", "r_squared_log_space"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fits)

    return fits


# =============================================================================
# CONCLUSIONS
# =============================================================================

def write_conclusions(results, diminishing_rows, scaling_fits):
    grouped = results_by_strategy(results)
    lines = []

    lines.append("# Core Component 3 — Data Scaling Conclusions")
    lines.append("")
    lines.append("## Fixed architecture")
    lines.append("")
    lines.append(
        f"All runs use the same ViT: blocks={core1.NUM_BLOCKS}, "
        f"heads={core1.NUM_HEADS}, embed_dim={core1.EMBED_DIM}, "
        f"MLP ratio={core1.MLP_RATIO}."
    )
    lines.append("")
    lines.append(
        "The validation split is unchanged for every experiment. Only the "
        "training subset and its sampling distribution change."
    )
    lines.append("")

    for strategy, rows in grouped.items():
        if not rows:
            continue

        lines.append(f"## {strategy.capitalize()} sampling")
        lines.append("")

        for row in rows:
            lines.append(
                f"- {row['data_percent']:.0f}% data "
                f"({row['num_training_samples']} samples): "
                f"best val accuracy={row['best_val_accuracy']:.4f}, "
                f"macro-F1={row['best_val_macro_f1']:.4f}, "
                f"training time={row['training_time_seconds']:.1f}s, "
                f"class coverage={100.0 * row['class_coverage']:.1f}%, "
                f"imbalance ratio={row['imbalance_ratio']:.3f}."
            )

        lines.append("")

    # Direct uniform vs nonuniform comparison at each scale.
    lines.append("## Uniform vs nonuniform")
    lines.append("")

    for fraction in DATA_FRACTIONS:
        matching = {
            row["strategy"]: row
            for row in results
            if abs(float(row["data_fraction"]) - fraction) < 1e-9
        }

        if "uniform" in matching and "nonuniform" in matching:
            uniform = matching["uniform"]
            nonuniform = matching["nonuniform"]

            accuracy_difference = (
                float(uniform["best_val_accuracy"])
                - float(nonuniform["best_val_accuracy"])
            )
            f1_difference = (
                float(uniform["best_val_macro_f1"])
                - float(nonuniform["best_val_macro_f1"])
            )

            lines.append(
                f"- {100 * fraction:.0f}%: uniform - nonuniform "
                f"accuracy = {accuracy_difference:+.4f}; "
                f"macro-F1 = {f1_difference:+.4f}."
            )

    lines.append("")
    lines.append(
        "At 100% unique-data coverage, uniform and nonuniform subsets contain "
        "the same full training set. Their sampling-distribution difference "
        "therefore disappears; this is a useful sanity check."
    )
    lines.append("")

    lines.append("## Diminishing returns")
    lines.append("")

    for row in diminishing_rows:
        lines.append(
            f"- {row['strategy']} {row['from_percent']:.0f}% -> "
            f"{row['to_percent']:.0f}%: added {row['additional_samples']} "
            f"samples, validation accuracy gain="
            f"{row['validation_accuracy_gain']:+.4f}, gain per 1000 added "
            f"samples={row['accuracy_gain_per_1000_added_samples']:+.4f}."
        )

    lines.append("")
    lines.append(
        "If the validation-accuracy gain per additional sample decreases as "
        "the dataset grows, the observed curve is exhibiting diminishing "
        "returns."
    )
    lines.append("")

    lines.append("## Empirical scaling-law diagnostic")
    lines.append("")

    if scaling_fits:
        for fit in scaling_fits:
            lines.append(
                f"- {fit['strategy']}: validation_error ~= "
                f"{fit['A']:.6g} * N^(-{fit['alpha']:.4f}), "
                f"log-space R^2={fit['r_squared_log_space']:.4f}."
            )

        lines.append("")
        lines.append(
            "A positive alpha means validation error tends to fall as more "
            "training examples are used. A high log-space R^2 means these four "
            "measured points are reasonably consistent with this simple "
            "power-law form. Four data scales are not enough to establish a "
            "universal scaling law, so interpret this as an empirical trend."
        )
    else:
        lines.append("Insufficient valid measurements for a power-law fit.")

    path = RESULTS_DIR / "conclusions.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# BUILD ALL DATA SUBSETS ONCE
# =============================================================================

def build_sampling_plan(train_samples, num_classes):
    uniform_orders = prepare_uniform_class_orders(train_samples, num_classes)
    nonuniform_order, nonuniform_class_weights = prepare_nonuniform_weighted_order(
        train_samples,
        num_classes,
    )

    plan = []

    for fraction in DATA_FRACTIONS:
        uniform_samples = uniform_stratified_subset(uniform_orders, fraction)
        plan.append({
            "strategy": "uniform",
            "fraction": fraction,
            "samples": uniform_samples,
            "class_weights": None,
        })

        nonuniform_samples = nonuniform_subset(nonuniform_order, fraction)
        plan.append({
            "strategy": "nonuniform",
            "fraction": fraction,
            "samples": nonuniform_samples,
            "class_weights": nonuniform_class_weights,
        })

    # Run smaller datasets first; within each scale, uniform then nonuniform.
    plan.sort(
        key=lambda item: (
            float(item["fraction"]),
            0 if item["strategy"] == "uniform" else 1,
        )
    )

    return plan


# =============================================================================
# MAIN
# =============================================================================

def run_core_component_3():
    log_path, log_file, original_stdout, original_stderr = start_persistent_logging()

    try:
        print("=" * 72)
        print("CORE COMPONENT 3 — DATA SCALING")
        print("=" * 72)
        print(f"Persistent log         : {log_path}")
        print("Model code             : UNCHANGED")
        print("Architecture           : FIXED BASELINE")
        print(f"Data fractions         : {DATA_FRACTIONS}")
        print(f"Sampling strategies    : {SAMPLING_STRATEGIES}")
        print(f"Fixed epochs           : {EXPERIMENT_EPOCHS}")
        print(f"Early stopping         : {USE_EARLY_STOPPING}")
        print("=" * 72)

        save_progress("initializing")

        # Exact same original split as Core Component 1.
        (
            class_names,
            full_train_samples,
            val_samples,
            train_class_counts,
        ) = core1.build_dataset(
            core1.TRAIN_PATH,
            val_fraction=core1.VAL_FRACTION,
            seed=core1.SEED,
        )

        num_classes = len(class_names)
        full_training_count = len(full_train_samples)

        print(f"Classes                : {num_classes}")
        print(f"Full training split    : {full_training_count}")
        print(f"Fixed validation split : {len(val_samples)}")
        print(
            f"Original class counts  : min={int(train_class_counts.min())}, "
            f"max={int(train_class_counts.max())}"
        )

        plan = build_sampling_plan(full_train_samples, num_classes)

        # Save all subset distributions before training starts.
        for item in plan:
            name = experiment_name(item["strategy"], item["fraction"])
            counts = subset_class_counts(item["samples"], num_classes)

            save_class_distribution(
                experiment=name,
                class_names=class_names,
                counts=counts,
                class_weights=item["class_weights"],
            )

        results = load_completed_results() if RESUME_COMPLETED_EXPERIMENTS else []
        completed_names = {result["experiment"] for result in results}

        if results:
            print(f"Resuming with {len(results)} completed experiment(s).")
            save_results_summary(results)

        for item in plan:
            strategy = item["strategy"]
            fraction = float(item["fraction"])
            name = experiment_name(strategy, fraction)

            if name in completed_names:
                print(f"Skipping completed experiment: {name}")
                continue

            train_subset = item["samples"]
            counts = subset_class_counts(train_subset, num_classes)

            result = train_one_experiment(
                experiment=name,
                strategy=strategy,
                fraction=fraction,
                train_subset=train_subset,
                val_samples=val_samples,
                class_names=class_names,
                full_training_count=full_training_count,
                class_counts=counts,
            )

            results.append(result)
            completed_names.add(name)

            # Persist the overall summary immediately after every completed run.
            save_results_summary(results)
            plot_experiment_history(name)

        # Final analysis once all requested runs available.
        results = load_completed_results()
        save_results_summary(results)

        for result in results:
            plot_experiment_history(result["experiment"])

        make_summary_plots(results)
        diminishing_rows = calculate_diminishing_returns(results)
        scaling_fits = analyze_scaling_law(results)
        conclusions_path = write_conclusions(
            results,
            diminishing_rows,
            scaling_fits,
        )

        save_progress(
            "complete",
            completed_experiments=len(results),
            conclusions=str(conclusions_path),
        )

        print()
        print("=" * 72)
        print("CORE COMPONENT 3 COMPLETE")
        print("=" * 72)
        print(f"Results summary        : {RESULTS_DIR / 'results_summary.csv'}")
        print(f"Diminishing returns    : {RESULTS_DIR / 'diminishing_returns.csv'}")
        print(f"Scaling-law fit        : {RESULTS_DIR / 'scaling_law_fit.csv'}")
        print(f"Conclusions            : {conclusions_path}")
        print(f"Plots                  : {PLOTS_DIR}")
        print(f"Checkpoints            : {CHECKPOINT_DIR}")
        print(f"Persistent run log     : {log_path}")
        print("=" * 72)

    except Exception as error:
        save_progress(
            "failed",
            error=str(error),
            traceback=traceback.format_exc(),
        )
        print()
        print("CORE COMPONENT 3 FAILED")
        traceback.print_exc()
        raise

    finally:
        stop_persistent_logging(
            log_file,
            original_stdout,
            original_stderr,
        )


if __name__ == "__main__":
    run_core_component_3()
