import os
import logging
from enum import Enum

# CONSTANTS #######################################
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Location where model file are saved.
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
NPY_EXT = ".npy"

# Location where grid plots to be stored.
PLOTS_DIR = os.path.join(ROOT_DIR, "saved_plots")

# Location where training data to be stored.
TRAINING_DIR = os.path.join(ROOT_DIR, "saved_train_inputs")

# LOGGINGS ########################################
logging.basicConfig(
    filename=os.path.join(ROOT_DIR, "logs", "logging.log"),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# HTTP CODES #######################################
class HTTPCode(Enum):
    OK = 200
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500