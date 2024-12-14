"""Default values for reference. Will be overwritten by config via hydra"""

import os

import torch

SEED = 1234

# data cache location
DESTINATION_PATH = "data/preprocessed/"

# Preprocessing args
AUDIO_EXT = ".wav"
LABEL_EXT = None
PROCESSED_EXT = ".npy"
SAMPLE_RATE = 16000

# Data loader args
BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count() // 2
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
