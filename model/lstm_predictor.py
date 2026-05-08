from pathlib import Path
import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model" / "weighted_lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"
LABELS_PATH = BASE_DIR / "model" / "label_order.pkl"

MAX_LEN = 120

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABELS_PATH, "rb") as f:
    ID_TO_LABEL = pickle.load(f)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_risk(text):
    cleaned_text = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = model.predict(padded, verbose=0)[0]

    low_prob = float(probs[0])
    medium_prob = float(probs[1])
    high_prob = float(probs[2])

    # Risk-sensitive decision rule, no keywords.
    # This is used because missing harmful content is more serious than false alarms.
    # Risk-sensitive decision rule, but not too aggressive
    if high_prob >= 0.50:
        risk = "HIGH"
        confidence = high_prob

    elif medium_prob >= 0.45 and medium_prob > low_prob:
        risk = "MEDIUM"
        confidence = medium_prob

    else:
        risk = "LOW"
        confidence = low_prob

    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "risk": risk,
        "confidence": confidence,
        "low_prob": low_prob,
        "medium_prob": medium_prob,
        "high_prob": high_prob,
        "mode": "Weighted LSTM"
    }
