from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "jigsaw.csv"

MODEL_PATH = BASE_DIR / "model" / "jigsaw_lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "model" / "jigsaw_lstm_tokenizer.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "model" / "jigsaw_lstm_label_encoder.pkl"


# ----------------------------
# Load Jigsaw dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Drop missing text
df = df.dropna(subset=["comment_text"])
df["comment_text"] = df["comment_text"].astype(str)


# ----------------------------
# Map Jigsaw labels to LOW / MEDIUM / HIGH
# ----------------------------
def map_label(row):
    if (
        row["severe_toxic"] == 1
        or row["threat"] == 1
        or row["insult"] == 1
        or row["obscene"] == 1
        or row["identity_hate"] == 1
    ):
        return "HIGH"
    elif row["toxic"] == 1:
        return "MEDIUM"
    else:
        return "LOW"


df["label"] = df.apply(map_label, axis=1)
df = df[["comment_text", "label"]]
df.columns = ["text", "label"]

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
max_words = 20000
max_len = 100

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
# Build LSTM model
# ----------------------------
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
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
    epochs=5,
    batch_size=64,
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

print("\n=== Jigsaw LSTM Accuracy ===")
print(accuracy)

print("\n=== Jigsaw LSTM Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Jigsaw LSTM Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))


# ----------------------------
# Save model + tokenizer + label encoder
# ----------------------------
model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("\nLSTM model saved to:", MODEL_PATH)
print("Tokenizer saved to:", TOKENIZER_PATH)
print("Label encoder saved to:", LABEL_ENCODER_PATH)
