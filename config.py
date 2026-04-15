# ================= PATHS ================= #
DATA_DIR   = "data/processed"
TRAIN_PATH = f"{DATA_DIR}/train"
VAL_PATH   = f"{DATA_DIR}/val"
OUTPUT_DIR = "outputs"

# ================= DATA SETTINGS ================= #
NUM_CLASSES = 2
NUM_FRAMES  = 16
IMG_SIZE    = 224

# ================= TRAINING SETTINGS ================= #
BATCH_SIZE  = 48
EPOCHS      = 80
NUM_WORKERS = 8

# ================= LEARNING ================= #
LR           = 3e-4
WEIGHT_DECAY = 1e-4

# ================= BACKBONE CONTROL ================= #
FREEZE_BACKBONE = True
UNFREEZE_EPOCH  = 5

# ================= REGULARIZATION ================= #
DROPOUT = 0.5

# ================= EARLY STOPPING ================= #
EARLY_STOPPING_PATIENCE = 12

# ================= ADVANCED SETTINGS ================= #
LABEL_SMOOTHING = 0.1
GRAD_CLIP       = 1.0

# ================= DATALOADER ================= #
PIN_MEMORY          = True
PERSISTENT_WORKERS  = True

# ================= DEVICE ================= #
DEVICE = "cuda"

# ================= LOGGING ================= #
PRINT_FREQ = 10

# ================= CHECKPOINT ================= #
SAVE_BEST_ONLY  = True
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best_model.pt"

# ================= STUDENT SETTINGS ================= #
# Student uses smaller input size to run on Jetson Nano
STUDENT_IMG_SIZE    = 112
STUDENT_NUM_FRAMES  = 16
STUDENT_BATCH_SIZE  = 32
STUDENT_EPOCHS      = 60
STUDENT_LR          = 1e-3
STUDENT_PATIENCE    = 15

# Distillation hyperparameters
DISTILL_TEMPERATURE = 4.0   # T — softens teacher probabilities
DISTILL_ALPHA       = 0.4   # weight on hard CE loss; (1-alpha) on soft KD loss

SOFT_LABELS_PATH    = f"{OUTPUT_DIR}/soft_labels_T4.pt"
STUDENT_MODEL_PATH  = f"{OUTPUT_DIR}/student_best.pt"

DEBUG = False
