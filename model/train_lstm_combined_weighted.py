import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
JIGSAW_PATH = BASE_DIR / "data" / "jigsaw.csv"
CUSTOM_PATH = BASE_DIR / "data" / "messages.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ROC_PATH = OUTPUT_DIR / "roc_auc_weighted_lstm.png"
CM_PATH = OUTPUT_DIR / "confusion_matrix_weighted_lstm.png"


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
df_custom = pd.read_csv(CUSTOM_PATH, encoding="utf-8")
df_custom.columns = [str(c).strip().lower() for c in df_custom.columns]
df_custom = df_custom.dropna(subset=["text", "label"])
df_custom["text"] = df_custom["text"].astype(str)
df_custom["label"] = df_custom["label"].astype(str).str.upper().str.strip()


# ----------------------------
# Combine datasets
# ----------------------------
df = pd.concat([df_jigsaw, df_custom], ignore_index=True)
df = df.dropna(subset=["text", "label"])

print("Combined label counts:")
print(df["label"].value_counts())


# ----------------------------
# Encode labels
# ----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

print("\nLabel mapping:")
for i, cls in enumerate(encoder.classes_):
    print(i, cls)


# ----------------------------
# Tokenization
# ----------------------------
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

X = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(X, maxlen=100)


# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------
# Class weights
# ----------------------------
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

print("\nClass weights:")
print(class_weights)


# ----------------------------
# Build LSTM model
# ----------------------------
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ----------------------------
# Train model
# ----------------------------
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)


# ----------------------------
# Evaluate
# ----------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=1)

print("\n=== Weighted Combined LSTM Accuracy ===")
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
cm = confusion_matrix(y_test, y_pred)

print("\n=== Confusion Matrix ===")
print(cm)


# ----------------------------
# Save confusion matrix graph
# ----------------------------
labels = encoder.classes_

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Weighted LSTM")
plt.colorbar()

plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(labels)), labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=300)
plt.close()

print("Confusion matrix saved at:", CM_PATH)


# ----------------------------
# ROC / AUC graph
# ----------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

plt.figure(figsize=(7, 5))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)

    class_name = encoder.classes_[i]
    plt.plot(fpr, tpr, label=f"{class_name} AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Weighted LSTM")
plt.legend()
plt.tight_layout()
plt.savefig(ROC_PATH, dpi=300)
plt.close()

print("ROC AUC graph saved at:", ROC_PATH)

# ----------------------------
# Save model + tokenizer + encoder
# ----------------------------
import pickle

MODEL_SAVE_PATH = BASE_DIR / "model" / "weighted_lstm_model.keras"
TOKENIZER_SAVE_PATH = BASE_DIR / "model" / "tokenizer.pkl"
ENCODER_SAVE_PATH = BASE_DIR / "model" / "label_encoder.pkl"

model.save(MODEL_SAVE_PATH)

with open(TOKENIZER_SAVE_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(encoder, f)

print("\nSaved model at:", MODEL_SAVE_PATH)
print("Saved tokenizer at:", TOKENIZER_SAVE_PATH)
print("Saved label encoder at:", ENCODER_SAVE_PATH)
