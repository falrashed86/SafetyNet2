from pathlib import Path
import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH ="model" / "weighted_lstm_model_fixed.h5"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"
LABELS_PATH = BASE_DIR / "model" / "label_order.pkl"

MAX_LEN = 120


model = load_model(MODEL_PATH, compile=False)

with open(TOKENIZER_PATH, "
