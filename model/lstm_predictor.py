import streamlit as st
import pickle
import re
from pathlib import Path

import numpy as np
from keras.models import load_model
from keras.initializers import Orthogonal
#import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences



BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model" / "weighted_lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"
LABELS_PATH = BASE_DIR / "model" / "label_order.pkl"

MAX_LEN = 120

model = None
tokenizer = None
ID_TO_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def load_assets():
    global model, tokenizer, ID_TO_LABEL

    if tokenizer is None:
        model=load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"Orthogonal":Orthogonal}
            )
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

    if LABELS_PATH.exists():
        with open(LABELS_PATH, "rb") as f:
            ID_TO_LABEL = pickle.load(f)

    if model is None:
        model = load_model(MODEL_PATH, compile=False)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_risk(text):
    cleaned_text = preprocess_text(text)

    try:
        load_assets()

        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        st.write("TEXT:",cleaned_text)
        st.write("SEQ:",seq)
        probs = model.predict(padded, verbose=0)[0]
        st.write("PROBS:",probs)
        low_prob = float(probs[0])
        medium_prob = float(probs[1])
        high_prob = float(probs[2])

        if high_prob >= 0.07:
            risk = "HIGH"
            confidence = high_prob
        elif medium_prob >= 0.20:
            risk = "MEDIUM"
            confidence = medium_prob
        else:
            risk = "LOW"
            confidence = low_prob

        

    except Exception as e:
        # emergency fallback so the Streamlit app still works
        low_prob = 0.0
        medium_prob = 0.0
        high_prob = 0.0
        confidence = 0.0
        risk = "MODEL ERROR:" + str(e)
       # print("MODEL ERROR",e)

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
