from pathlib import Path
import math
import random

import numpy as np
import cupy as cp

from PIL import Image, ImageEnhance, ImageOps

from model.vit_architecture import VisionTransformer


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

TRAIN_PATH = SCRIPT_DIR / 'train'

CHECKPOINT_DIR = SCRIPT_DIR.parent / 'checkpoints'

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / 'best_vit_checkpoint_improved.npz'

NORMALIZATION_CACHE_PATH = CHECKPOINT_DIR / 'training_rgb_stats.npz'


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 64
PATCH_SIZE = 16
CHANNELS = 3

EMBED_DIM = 256
NUM_HEADS = 4

# forward_model() loops from 0 to NUM_BLOCKS-1.
# backward_model() loops in reverse.
NUM_BLOCKS = 3
MLP_RATIO = 4

BATCH_SIZE = 32

# Give the better optimizer/schedule enough time.
EPOCHS = 20

VAL_FRACTION = 0.20
SEED = 42


# ============================================================
# ADAMW CONFIGURATION
# ============================================================

MAX_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-5

WARMUP_EPOCHS = 5

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

WEIGHT_DECAY = 0.10


# ============================================================
# DATA AUGMENTATION
# ============================================================

USE_AUGMENTATION = True

# Epoch 1: 100% non-augmented.
# Epoch 2 onward: ~20% non-augmented, ~80% augmented.
# Non-augmented still means resize/letterbox + [0,1] + RGB normalization.
FIRST_EPOCH_ORIGINAL_ONLY = True
ORIGINAL_IMAGE_PROB = 0.20

HORIZONTAL_FLIP_PROB = 0.5

# Factor is sampled from:
#   [1 - COLOR_JITTER, 1 + COLOR_JITTER]
#
# 0.15 means roughly 0.85 ... 1.15.
COLOR_JITTER = 0.20

# Resize into a slightly larger square and then randomly crop
# back to IMAGE_SIZE x IMAGE_SIZE. Aspect ratio of the original image itself
# is preserved while resizing.
ZOOM_MIN = 1.00
ZOOM_MAX = 1.15


# ============================================================
# ANTI-OVERFITTING REGULARIZATION
# ============================================================

# Transformer residual-branch dropout.
DROPOUT_RATE = 0.10

# Embedding dropout.
#
# Applied inside the model after:
#     CLS token + positional embedding
#
# and before:
#     Transformer Block 0
EMBEDDING_DROPOUT_RATE = 0.10

# Label smoothing:
# correct class target becomes approximately 0.90 instead of 1.00.
LABEL_SMOOTHING = 0.10

# Random erasing is applied only to training images,
# after RGB normalization.
RANDOM_ERASE_PROB = 0.15
RANDOM_ERASE_MIN_AREA = 0.02
RANDOM_ERASE_MAX_AREA = 0.10
RANDOM_ERASE_MIN_ASPECT = 0.50
RANDOM_ERASE_MAX_ASPECT = 2.00

# Stop when validation loss has failed to improve for this
# many consecutive epochs.
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-3


# ============================================================
# NORMALIZATION
# ============================================================

# If False and a compatible cache exists, reuse it.
RECOMPUTE_NORMALIZATION_STATS = False

NORMALIZATION_EPS = 1e-6


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

cp.random.seed(SEED)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


# ============================================================
# DATASET DISCOVERY
# ============================================================

def get_class_folders(root):
    if not root.exists():
        raise FileNotFoundError(f'Training directory not found: {root}')

    folders = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)

    if len(folders) == 0:
        raise RuntimeError(f'No class folders found in {root}')

    return folders


def get_images_in_class(class_folder):
    return sorted([path for path in class_folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS])


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

def build_dataset(root, val_fraction=0.2, seed=42):
    """
    Split every class independently.

    Validation data never enters the training list.
    """

    class_folders = get_class_folders(root)

    class_names = [folder.name for folder in class_folders]

    num_classes = len(class_names)

    train_samples = []
    val_samples = []

    train_class_counts = np.zeros(num_classes, dtype=np.int64)

    for (class_idx, class_folder) in enumerate(class_folders):
        image_paths = get_images_in_class(class_folder)

        if len(image_paths) == 0:
            raise RuntimeError(f'No images in class folder: {class_folder}')

        class_rng = random.Random(seed + class_idx)

        image_paths = image_paths.copy()

        class_rng.shuffle(image_paths)

        if len(image_paths) == 1:
            num_val = 0

        else:

            num_val = int(round(len(image_paths) * val_fraction))

            num_val = max(1, num_val)

            num_val = min(num_val, len(image_paths) - 1)

        val_paths = image_paths[:num_val]

        train_paths = image_paths[num_val:]

        train_class_counts[class_idx] = len(train_paths)

        for image_path in train_paths:
            train_samples.append((image_path, class_idx))

        for image_path in val_paths:
            val_samples.append((image_path, class_idx))

    return (class_names, train_samples, val_samples, train_class_counts)


# ============================================================
# SHUFFLED MINI-BATCHES
# ============================================================

def create_epoch_batches(train_samples, batch_size, rng):
    """
    Shuffle all training indices once.

    Then consume the shuffled order sequentially in batches.
    """

    indices = list(range(len(train_samples)))

    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        yield [train_samples[idx] for idx in batch_indices]


# ============================================================
# ASPECT-RATIO-PRESERVING RESIZE
# ============================================================

def letterbox_image(image, canvas_size):
    """
    Resize image to fit inside a square canvas without stretching.

    The original aspect ratio is preserved.
    """

    image = image.convert('RGB')

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions.')

    scale = min(canvas_size / width, canvas_size / height)

    new_width = max(1, int(round(width * scale)))

    new_height = max(1, int(round(height * scale)))

    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    canvas = Image.new('RGB', (canvas_size, canvas_size), color=(0, 0, 0))

    left = (canvas_size - new_width) // 2

    top = (canvas_size - new_height) // 2

    canvas.paste(resized, (left, top))

    return canvas


def deterministic_preprocess_pil(image):
    """
    Used by:
        validation
        normalization-statistics calculation
        later test.py

    Always returns exactly IMAGE_SIZE x IMAGE_SIZE.
    """

    return letterbox_image(image, IMAGE_SIZE)


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def augment_training_pil(image, rng):
    """
    Mild augmentations only.

    1. Random horizontal flip.
    2. Mild color jitter.
    3. Mild zoom/translation while keeping image aspect ratio.
    """

    image = image.convert('RGB')

    # --------------------------------------------------------
    # Horizontal flip.
    # --------------------------------------------------------

    if rng.random() < HORIZONTAL_FLIP_PROB:
        image = ImageOps.mirror(image)

    # --------------------------------------------------------
    # Color jitter.
    # --------------------------------------------------------

    if COLOR_JITTER > 0:
        brightness_factor = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)

        contrast_factor = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)

        saturation_factor = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)

        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        image = ImageEnhance.Color(image).enhance(saturation_factor)

    # --------------------------------------------------------
    # Mild zoom.
    #
    # First letterbox into a slightly larger square.
    # Then randomly crop IMAGE_SIZE x IMAGE_SIZE.
    #
    # The resize itself still preserves original aspect ratio.
    # --------------------------------------------------------

    zoom = rng.uniform(ZOOM_MIN, ZOOM_MAX)

    enlarged_size = max(IMAGE_SIZE, int(round(IMAGE_SIZE * zoom)))

    image = letterbox_image(image, enlarged_size)

    if enlarged_size > IMAGE_SIZE:
        max_offset = enlarged_size - IMAGE_SIZE

        left = rng.randint(0, max_offset)

        top = rng.randint(0, max_offset)

        image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    return image


# ============================================================
# RAW IMAGE -> [0,1]
# ============================================================

def pil_to_float_array(image):
    array = np.asarray(image, dtype=np.float32).copy()

    array *= np.float32(1.0 / 255.0)

    return array


# ============================================================
# TRAINING-SET RGB MEAN / STD
# ============================================================

def compute_training_mean_std(train_samples):
    """
    Calculate normalization statistics from TRAINING images only.

    Validation images are not used.

    Uses deterministic letterbox preprocessing and streams over
    images, so the whole dataset is never stored in RAM.
    """

    channel_sum = np.zeros(3, dtype=np.float64)

    channel_square_sum = np.zeros(3, dtype=np.float64)

    total_pixels = 0

    total_images = len(train_samples)

    print('Computing training RGB mean/std...')

    for (index, (image_path, _)) in enumerate(train_samples, start=1):
        with Image.open(image_path) as image:
            image = deterministic_preprocess_pil(image)

            array = pil_to_float_array(image)

        flat = array.reshape(-1, 3)

        channel_sum += flat.sum(axis=0, dtype=np.float64)

        channel_square_sum += (flat.astype(np.float64) ** 2).sum(axis=0)

        total_pixels += flat.shape[0]

        if index % 2000 == 0 or index == total_images:
            print(f'  stats images: {index}/{total_images}')

    mean = channel_sum / total_pixels

    variance = channel_square_sum / total_pixels - mean ** 2

    variance = np.maximum(variance, NORMALIZATION_EPS)

    std = np.sqrt(variance)

    return (mean.astype(np.float32), std.astype(np.float32))


def get_training_mean_std(train_samples):
    """
    Reuse cached stats when possible to avoid decoding the
    entire training set every run.
    """

    if not RECOMPUTE_NORMALIZATION_STATS and NORMALIZATION_CACHE_PATH.exists():
        cache = np.load(NORMALIZATION_CACHE_PATH, allow_pickle=False)

        cached_count = int(cache['num_train'])

        cached_seed = int(cache['seed'])

        if cached_count == len(train_samples) and cached_seed == SEED:
            mean = cache['mean'].astype(np.float32)

            std = cache['std'].astype(np.float32)

            print('Loaded cached RGB normalization statistics.')

            return (mean, std)

    mean, std = compute_training_mean_std(train_samples)

    np.savez(NORMALIZATION_CACHE_PATH, mean=mean, std=std, num_train=np.asarray(len(train_samples), dtype=np.int64), seed=np.asarray(SEED, dtype=np.int64))

    print('Saved RGB normalization statistics:')

    print(f'  {NORMALIZATION_CACHE_PATH}')

    return (mean, std)


# ============================================================
# RANDOM ERASING
# ============================================================

def random_erasing(array, rng):
    """
    Randomly erase a small rectangle in a TRAINING image.

    array is already normalized.

    Filling with 0 means filling with the dataset mean in
    normalized coordinates.
    """

    if rng is None or rng.random() >= RANDOM_ERASE_PROB:
        return array

    height, width, _ = array.shape

    image_area = height * width

    for _ in range(10):
        target_area = image_area * rng.uniform(RANDOM_ERASE_MIN_AREA, RANDOM_ERASE_MAX_AREA)

        aspect_ratio = rng.uniform(RANDOM_ERASE_MIN_ASPECT, RANDOM_ERASE_MAX_ASPECT)

        erase_height = int(round(math.sqrt(target_area / aspect_ratio)))

        erase_width = int(round(math.sqrt(target_area * aspect_ratio)))

        if 0 < erase_height <= height and 0 < erase_width <= width:
            top = rng.randint(0, height - erase_height)

            left = rng.randint(0, width - erase_width)

            array[top:top + erase_height, left:left + erase_width, :] = np.float32(0.0)

            break

    return array


# ============================================================
# LOAD ONE IMAGE
# ============================================================

def load_one_image(image_path, mean, std, training=False, rng=None, epoch=None):
    """
    Load and preprocess one image.

    Epoch 1:
        100% non-augmented training images.

    Epoch 2 onward:
        ~20% non-augmented and ~80% randomly augmented.

    A non-augmented image still receives deterministic resize/letterbox,
    [0,1] conversion, and RGB normalization. It only skips random
    flip/color-jitter/zoom/random-erasing.

    Validation is always deterministic and augmentation-free.
    """

    use_augmented_version = False

    if training and USE_AUGMENTATION:
        if rng is None:
            raise ValueError("Training augmentation requires rng.")

        if epoch is None:
            raise ValueError("Training preprocessing requires the current epoch.")

        if FIRST_EPOCH_ORIGINAL_ONLY and epoch == 1:
            use_augmented_version = False
        else:
            # Each image independently has a 20% chance of staying
            # non-augmented and an 80% chance of augmentation.
            use_augmented_version = rng.random() >= ORIGINAL_IMAGE_PROB

    with Image.open(image_path) as image:
        if use_augmented_version:
            image = augment_training_pil(image, rng)
        else:
            image = deterministic_preprocess_pil(image)

        array = pil_to_float_array(image)

    # RGB normalization is applied to every image.
    array = array - mean[None, None, :]

    array = array / (std[None, None, :] + np.float32(NORMALIZATION_EPS))

    # Random erasing is an augmentation, so do it only for the
    # images selected for the augmented path.
    if use_augmented_version:
        array = random_erasing(array, rng)

    return array.astype(np.float32, copy=False)


# ============================================================
# LOAD BATCH TO GPU
# ============================================================

def load_batch_to_gpu(samples, mean, std, training=False, rng=None, epoch=None):
    """
    CPU:
        decode -> preprocess -> normalize

    Then one transfer to GPU.

    Preallocation avoids the old np.stack memory spike.
    """

    B = len(samples)

    images_cpu = np.empty((B, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)

    labels_cpu = np.empty(B, dtype=np.int32)

    for (index, (image_path, label)) in enumerate(samples):
        images_cpu[index] = load_one_image(image_path, mean, std, training=training, rng=rng, epoch=epoch)

        labels_cpu[index] = label

    images_gpu = cp.asarray(images_cpu)

    labels_gpu = cp.asarray(labels_cpu)

    return (images_gpu, labels_gpu)


# ============================================================
# ORDINARY SOFTMAX CROSS ENTROPY
# ============================================================

def cross_entropy_forward_backward(logits, labels, label_smoothing=LABEL_SMOOTHING):
    """
    Label-smoothed softmax cross entropy.

    We return:
        objective_loss:
            smoothed CE used for backprop.

        hard_ce_loss:
            ordinary CE, useful for comparing train loss with
            validation loss.

        dlogits:
            gradient for backprop.

        predictions:
            argmax class prediction.
    """

    B, C = logits.shape

    shifted_logits = logits - cp.max(logits, axis=1, keepdims=True)

    exp_logits = cp.exp(shifted_logits)

    probabilities = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)

    correct_probabilities = probabilities[cp.arange(B), labels]

    hard_ce_loss = cp.mean(-cp.log(correct_probabilities + cp.float32(1e-12)))

    smoothing = cp.float32(label_smoothing)

    targets = cp.full((B, C), smoothing / cp.float32(C), dtype=cp.float32)

    targets[cp.arange(B), labels] += cp.float32(1.0) - smoothing

    objective_loss = cp.mean(-cp.sum(targets * cp.log(probabilities + cp.float32(1e-12)), axis=1))

    dlogits = probabilities - targets

    dlogits /= cp.float32(B)

    predictions = cp.argmax(probabilities, axis=1)

    return (objective_loss, hard_ce_loss, dlogits, predictions)

# ============================================================
# WARMUP + COSINE LEARNING RATE
# ============================================================

def get_learning_rate(global_step, total_steps, warmup_steps):
    """
    Warmup:
        approximately 0 -> MAX_LEARNING_RATE

    Then:
        cosine decay
        MAX_LEARNING_RATE -> MIN_LEARNING_RATE
    """

    if warmup_steps > 0 and global_step < warmup_steps:
        warmup_fraction = (global_step + 1) / warmup_steps

        return MAX_LEARNING_RATE * warmup_fraction

    remaining_steps = max(1, total_steps - warmup_steps)

    progress = (global_step - warmup_steps) / remaining_steps

    progress = min(1.0, max(0.0, progress))

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return MIN_LEARNING_RATE + (MAX_LEARNING_RATE - MIN_LEARNING_RATE) * cosine


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, val_samples, mean, std):
    """
    Validation:
        deterministic preprocessing
        no augmentation
        ordinary cross entropy
        no parameter updates
    """

    if len(val_samples) == 0:
        return (float('nan'), float('nan'))

    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    for start in range(0, len(val_samples), BATCH_SIZE):
        batch_samples = val_samples[start:start + BATCH_SIZE]

        images, labels = load_batch_to_gpu(batch_samples, mean, std, training=False, rng=None, epoch=None)

        logits = model.forward(images, training=False)

        B = labels.shape[0]

        shifted_logits = logits - cp.max(logits, axis=1, keepdims=True)

        exp_logits = cp.exp(shifted_logits)

        probabilities = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)

        correct_probabilities = probabilities[cp.arange(B), labels]

        batch_loss_sum = cp.sum(-cp.log(correct_probabilities + cp.float32(1e-12)))

        predictions = cp.argmax(probabilities, axis=1)

        total_loss_sum += float(batch_loss_sum.get())

        total_correct += int(cp.sum(predictions == labels).get())

        total_samples += B

    val_loss = total_loss_sum / total_samples

    val_accuracy = total_correct / total_samples

    return (val_loss, val_accuracy)


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(model, class_names, epoch, val_accuracy, mean, std):
    state = {
        "epoch":
            np.asarray(
                epoch,
                dtype=np.int32,
            ),

        "val_accuracy":
            np.asarray(
                val_accuracy,
                dtype=np.float32,
            ),

        "class_names":
            np.asarray(
                class_names
            ),

        # Important for test.py.
        "rgb_mean":
            np.asarray(
                mean,
                dtype=np.float32,
            ),

        "rgb_std":
            np.asarray(
                std,
                dtype=np.float32,
            ),

        "image_size":
            np.asarray(
                IMAGE_SIZE,
                dtype=np.int32,
            ),

        "patch_size":
            np.asarray(
                PATCH_SIZE,
                dtype=np.int32,
            ),

        "embed_dim":
            np.asarray(
                EMBED_DIM,
                dtype=np.int32,
            ),

        "num_heads":
            np.asarray(
                NUM_HEADS,
                dtype=np.int32,
            ),

        "num_blocks":
            np.asarray(
                NUM_BLOCKS,
                dtype=np.int32,
            ),

        "mlp_ratio":
            np.asarray(
                MLP_RATIO,
                dtype=np.int32,
            ),

        "max_learning_rate":
            np.asarray(
                MAX_LEARNING_RATE,
                dtype=np.float32,
            ),

        "min_learning_rate":
            np.asarray(
                MIN_LEARNING_RATE,
                dtype=np.float32,
            ),

        "warmup_epochs":
            np.asarray(
                WARMUP_EPOCHS,
                dtype=np.int32,
            ),

        "weight_decay":
            np.asarray(
                WEIGHT_DECAY,
                dtype=np.float32,
            ),

        "dropout_rate":
            np.asarray(
                DROPOUT_RATE,
                dtype=np.float32,
            ),

        "embedding_dropout_rate":
            np.asarray(
                EMBEDDING_DROPOUT_RATE,
                dtype=np.float32,
            ),

        "label_smoothing":
            np.asarray(
                LABEL_SMOOTHING,
                dtype=np.float32,
            ),

        "first_epoch_original_only":
            np.asarray(
                FIRST_EPOCH_ORIGINAL_ONLY,
                dtype=np.bool_,
            ),

        "original_image_prob":
            np.asarray(
                ORIGINAL_IMAGE_PROB,
                dtype=np.float32,
            ),
    }

    for (name, value) in model.state_dict().items():
        state[name] = cp.asnumpy(value)

    np.savez(CHECKPOINT_PATH, **state)


# ============================================================
# CONFIGURATION PRINT
# ============================================================

def print_configuration(class_names, train_class_counts, num_train, num_val, mean, std, steps_per_epoch):
    print('================================================')

    print('IMPROVED ViT TRAINING CONFIGURATION')

    print('================================================')

    print(f'Classes              : {len(class_names)}')

    print(f'Training images      : {num_train}')

    print(f'Validation images    : {num_val}')

    print(f'Smallest class       : {int(train_class_counts.min())}')

    print(f'Largest class        : {int(train_class_counts.max())}')

    print(f'Imbalance ratio      : {float(train_class_counts.max()) / float(train_class_counts.min()):.3f}')

    print('Loss                 : ordinary cross entropy')

    print(f'Batch size           : {BATCH_SIZE}')

    print(f'Steps / epoch        : {steps_per_epoch}')

    print(f'Epochs               : {EPOCHS}')

    print(f'Blocks               : {NUM_BLOCKS}')

    print(f'Embedding dim        : {EMBED_DIM}')

    print(f'Heads                : {NUM_HEADS}')

    print(f'Optimizer            : AdamW')

    print(f'Max LR               : {MAX_LEARNING_RATE}')

    print(f'Min LR               : {MIN_LEARNING_RATE}')

    print(f'Warmup epochs        : {WARMUP_EPOCHS}')

    print(f'Weight decay         : {WEIGHT_DECAY}')

    print(f'Block dropout        : {DROPOUT_RATE}')

    print(f'Embedding dropout    : {EMBEDDING_DROPOUT_RATE}')

    print(f'Label smoothing      : {LABEL_SMOOTHING}')

    print(f'Random erase prob    : {RANDOM_ERASE_PROB}')

    print(f'Early-stop patience  : {EARLY_STOPPING_PATIENCE}')

    print(f'Augmentation         : {USE_AUGMENTATION}')

    if USE_AUGMENTATION:
        print('Epoch 1 images       : 100% non-augmented')
        print(f'Epoch 2+ images      : ~{ORIGINAL_IMAGE_PROB * 100:.0f}% non-augmented / ~{(1.0 - ORIGINAL_IMAGE_PROB) * 100:.0f}% augmented')

    print('RGB mean             : ' + np.array2string(mean, precision=6))

    print('RGB std              : ' + np.array2string(std, precision=6))

    print(f'Checkpoint           : {CHECKPOINT_PATH}')

    print('================================================')

    print()


# ============================================================
# TRAIN
# ============================================================

def train():

    # --------------------------------------------------------
    # 1. Dataset split.
    # --------------------------------------------------------

    class_names, train_samples, val_samples, train_class_counts = build_dataset(TRAIN_PATH, val_fraction=VAL_FRACTION, seed=SEED)

    num_classes = len(class_names)

    num_train = len(train_samples)

    num_val = len(val_samples)

    steps_per_epoch = math.ceil(num_train / BATCH_SIZE)

    total_steps = EPOCHS * steps_per_epoch

    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    # --------------------------------------------------------
    # 2. Training-only RGB mean/std.
    # --------------------------------------------------------

    mean, std = get_training_mean_std(train_samples)

    print_configuration(class_names, train_class_counts, num_train, num_val, mean, std, steps_per_epoch)

    # --------------------------------------------------------
    # 3. Model.
    # --------------------------------------------------------

    model = VisionTransformer(
        num_classes=num_classes,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        mlp_ratio=MLP_RATIO,
        eps=1e-5,
        dropout_rate=DROPOUT_RATE,
        embedding_dropout_rate=EMBEDDING_DROPOUT_RATE,
    )

    # Separate RNGs:
    #
    # shuffle_rng controls sample order.
    # augmentation_rng controls random augmentation.
    shuffle_rng = random.Random(SEED)

    augmentation_rng = random.Random(SEED + 100000)

    best_val_accuracy = -1.0

    best_val_loss = float('inf')

    epochs_without_val_loss_improvement = 0

    global_step = 0

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(1, EPOCHS + 1):
        if USE_AUGMENTATION and FIRST_EPOCH_ORIGINAL_ONLY and epoch == 1:
            print(f'Epoch {epoch:02d}: 100% non-augmented training images.')
        elif USE_AUGMENTATION:
            print(f'Epoch {epoch:02d}: ~{ORIGINAL_IMAGE_PROB * 100:.0f}% non-augmented + ~{(1.0 - ORIGINAL_IMAGE_PROB) * 100:.0f}% augmented.')
        else:
            print(f'Epoch {epoch:02d}: augmentation disabled.')

        epoch_loss_sum = 0.0
        epoch_objective_sum = 0.0

        epoch_correct = 0
        epoch_seen = 0

        batch_iterator = create_epoch_batches(train_samples, BATCH_SIZE, shuffle_rng)

        # ====================================================
        # MINI-BATCH LOOP
        # ====================================================

        for (step, batch_samples) in enumerate(batch_iterator, start=1):

            # ------------------------------------------------
            # A. Load + augment + normalize.
            # ------------------------------------------------

            images, labels = load_batch_to_gpu(batch_samples, mean, std, training=True, rng=augmentation_rng, epoch=epoch)

            # ------------------------------------------------
            # B. Complete ViT forward pass.
            # ------------------------------------------------

            logits = model.forward(images, training=True)

            # ------------------------------------------------
            # C. Ordinary cross entropy.
            # ------------------------------------------------

            loss, hard_ce_loss, dlogits, predictions = cross_entropy_forward_backward(logits, labels, label_smoothing=LABEL_SMOOTHING)

            # ------------------------------------------------
            # D. Complete ViT backward pass.
            # ------------------------------------------------

            model.backward(dlogits)

            # ------------------------------------------------
            # E. Warmup + cosine LR.
            # ------------------------------------------------

            current_lr = get_learning_rate(global_step, total_steps, warmup_steps)

            # ------------------------------------------------
            # F. Manual AdamW update.
            # ------------------------------------------------

            model.adamw_step(learning_rate=current_lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, adam_eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

            global_step += 1

            # ------------------------------------------------
            # G. Metrics.
            # ------------------------------------------------

            B = labels.shape[0]

            batch_correct = int(cp.sum(predictions == labels).get())

            # loss is the mean CE for this mini-batch.
            # Multiply by B so epoch loss becomes the mean over
            # all individual training images.
            # Ordinary CE remains comparable with validation loss.
            epoch_loss_sum += float(hard_ce_loss.get()) * B

            # Smoothed loss is the actual optimization objective.
            epoch_objective_sum += float(loss.get()) * B

            epoch_correct += batch_correct

            epoch_seen += B

            if step == 1 or step % 10 == 0 or step == steps_per_epoch:
                batch_accuracy = batch_correct / B

                print(
                    f"Epoch "
                    f"{epoch:02d}/{EPOCHS:02d} | "
                    f"Step "
                    f"{step:04d}/{steps_per_epoch:04d} | "
                    f"LR "
                    f"{current_lr:.6e} | "
                    f"CE "
                    f"{float(hard_ce_loss.get()):.4f} | "
                    f"Smooth "
                    f"{float(loss.get()):.4f} | "
                    f"Batch Acc "
                    f"{batch_accuracy:.4f}"
                )

        # ====================================================
        # END OF EPOCH
        # ====================================================

        train_loss = epoch_loss_sum / epoch_seen

        train_objective = epoch_objective_sum / epoch_seen

        train_accuracy = epoch_correct / epoch_seen

        val_loss, val_accuracy = evaluate(model, val_samples, mean, std)

        print()

        print(f'Epoch {epoch:02d} summary')

        print(f'Train Loss     : {train_loss:.4f}')

        print(f'Train Objective: {train_objective:.4f}')

        print(f'Train Accuracy : {train_accuracy:.4f}')

        if not math.isnan(val_loss):
            print(f'Val Loss       : {val_loss:.4f}')

            print(f'Val Accuracy   : {val_accuracy:.4f}')

        # ----------------------------------------------------
        # Save best validation checkpoint.
        # ----------------------------------------------------

        if not math.isnan(val_accuracy) and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            save_checkpoint(model, class_names, epoch, val_accuracy, mean, std)

            print('Saved new best checkpoint:')

            print(f'  {CHECKPOINT_PATH}')

        if not math.isnan(val_accuracy):
            generalization_gap = train_accuracy - val_accuracy

            print(f'Accuracy Gap   : {generalization_gap:.4f}')

        # ----------------------------------------------------
        # Early stopping based on validation loss.
        # ----------------------------------------------------

        if not math.isnan(val_loss) and val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss

            epochs_without_val_loss_improvement = 0

        else:
            epochs_without_val_loss_improvement += 1

            print(f'Val loss did not improve | patience {epochs_without_val_loss_improvement}/{EARLY_STOPPING_PATIENCE}')

        print('------------------------------------------------')

        if epochs_without_val_loss_improvement >= EARLY_STOPPING_PATIENCE:
            print('Early stopping: validation loss has stopped improving.')
            break

    print()

    print('Training complete.')

    if best_val_accuracy >= 0:
        print(f'Best validation accuracy: {best_val_accuracy:.4f}')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    train()
