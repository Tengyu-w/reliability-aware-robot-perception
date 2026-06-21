"""Project configuration.

The defaults are intentionally lightweight so the project can run as a
research prototype. Override paths from the command line or with environment
variables when running on a new machine.
"""

import os
import torch
import warnings

warnings.filterwarnings('ignore')

# ==================== Path Configuration ====================
# These can be overridden with environment variables or CLI args in main.py.
TRAIN_DATA_PATH = os.getenv("VIDEO_TRAIN_DATA_PATH", "/media/ori/5A1280CB1280AD97/sensing/train_set")
TEST_DATA_PATH = os.getenv("VIDEO_TEST_DATA_PATH", "/media/ori/5A1280CB1280AD97/sensing/test_set")
LABEL_CSV = os.getenv("VIDEO_LABEL_CSV", "/media/ori/5A1280CB1280AD97/sensing/train_set_labels.csv")
OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "outputs")

# ==================== Hyperparameter Settings ====================
IMG_SIZE = 224
NUM_FRAMES = 16
BATCH_SIZE = 8  # Reduced batch size
EPOCHS = 1 # Reduced number of training epochs
LEARNING_RATE = 1e-4 # Using fixed learning rate
HIDDEN_DIM = 256  # Reduced hidden layer dimension
LSTM_LAYERS = 1  # Reduced number of LSTM layers
BIDIRECTIONAL = False  # Do not use bidirectional LSTM (simplified)
DROP_RATE = 0.3
SEED = 42
VAL_SPLIT = 0.2
NUM_WORKERS = 2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
