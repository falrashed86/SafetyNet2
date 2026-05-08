import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
JIGSAW_PATH = BASE_DIR / "data" / "jigsaw.csv"
CUSTOM_PATH = BASE_DIR / "data" / "messages.csv"


# ----------------------------
# Load Jigsaw dataset
# ----------------------------
df_jigsaw = pd.read_csv(JIGSAW_PATH)

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

df_jigsaw["label"] = df_jigsaw.apply(map_label, axis=1)
df_jigsaw = df_jigsaw[["comment_text", "label"]]
df_jigsaw.columns = ["text", "label"]


# ----------------------------
# Load custom dataset
# ----------------------------
df_custom = pd.read_csv(CUSTOM_PATH)
df_custom.columns = [c.lower().strip() for c in df_custom.columns]
df_custom = df_custom.dropna(subset=["text", "label"])
df_custom["text"] = df_custom["text"].astype(str)
df_custom["label"] = df_custom["label"].astype(str).str.upper().str.strip()


# ----------------------------
# Combine datasets
# ----------------------------
df = pd.concat([df_jigsaw, df_custom], ignore_index=True)

print("Combined label counts:")
print(df["label"].value_counts())


# ----------------------------
# Encode labels
# ----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])


# ----------------------------
# Tokenization
# ----------------------------
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

X = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(X, maxlen=100)


# ----------------------------
# Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------
# Model
# ----------------------------
model = Sequential([
    Embedding(10000, 128, input_length=100),
    LSTM(64),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# ----------------------------
# Train
# ----------------------------
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    verbose=1
)


# ----------------------------
# Evaluate accuracy
# ----------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print("\n=== Combined LSTM Accuracy ===")
print("Accuracy:", acc)


# ----------------------------
# Predict
# ----------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


# ----------------------------
# Classification report
# ----------------------------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))


# ----------------------------
# Confusion matrix
# ----------------------------
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
