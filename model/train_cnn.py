from pathlib import Path
import re
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "messages.csv"

TOKENIZER_PATH = BASE_DIR / "model" / "cnn_tokenizer.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "model" / "cnn_label_encoder.pkl"
CNN_MODEL_PATH = BASE_DIR / "model" / "cnn_model.keras"


# ----------------------------
# Simple preprocessing
# Keep it same style as LSTM
# ----------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df.columns = [str(c).strip().lower() for c in df.columns]

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).apply(preprocess_text)
df["label"] = df["label"].astype(str).str.upper().str.strip()

print("Dataset loaded successfully.")
print(df.head())
print("\nLabel counts:")
print(df["label"].value_counts())


# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# ----------------------------
# Tokenization
# ----------------------------
max_words = 5000
max_len = 30

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")


# ----------------------------
# Encode labels
# ----------------------------
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)

y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)

print("\nClasses:")
print(label_encoder.classes_)


# ----------------------------
# Build CNN model
# ----------------------------
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    Conv1D(filters=64, kernel_size=3, activation="relu"),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ----------------------------
# Train
# ----------------------------
history = model.fit(
    X_train_pad,
    y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=8,
    verbose=1
)


# ----------------------------
# Predict
# ----------------------------
y_pred_probs = model.predict(X_test_pad)
y_pred_enc = np.argmax(y_pred_probs, axis=1)
y_pred = label_encoder.inverse_transform(y_pred_enc)


# ----------------------------
# Evaluate
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n=== CNN Accuracy ===")
print(accuracy)

print("\n=== CNN Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== CNN Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))


# ----------------------------
# Save model + tokenizer + label encoder
# ----------------------------
model.save(CNN_MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("\nCNN model saved to:", CNN_MODEL_PATH)
print("Tokenizer saved to:", TOKENIZER_PATH)
print("Label encoder saved to:", LABEL_ENCODER_PATH)
