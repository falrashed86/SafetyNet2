from pathlib import Path
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

JIGSAW_PATH = DATA_DIR / "jigsaw.csv"
CUSTOM_PATH = DATA_DIR / "messages.csv"

MODEL_PATH = MODEL_DIR / "weighted_lstm_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
LABELS_PATH = MODEL_DIR / "label_order.pkl"

CM_PATH = OUTPUT_DIR / "confusion_matrix_weighted_lstm.png"
ROC_PATH = OUTPUT_DIR / "roc_auc_weighted_lstm.png"


# ----------------------------
# Settings
# ----------------------------
MAX_WORDS = 20000
MAX_LEN = 120

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]
LABEL_TO_ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ID_TO_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Load Jigsaw
# ----------------------------
def load_jigsaw():
    df = pd.read_csv(JIGSAW_PATH)
    df = df.dropna(subset=["comment_text"])

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
    return df


# ----------------------------
# Load custom
# ----------------------------
def load_custom():
    df = pd.read_csv(CUSTOM_PATH, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(LABEL_ORDER)]
    return df[["text", "label"]]


# ----------------------------
# Load and combine
# ----------------------------
jigsaw_df = load_jigsaw()
custom_df = load_custom()

df = pd.concat([jigsaw_df, custom_df], ignore_index=True)
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(str).str.upper().str.strip()
df = df[df["label"].isin(LABEL_ORDER)]

df["cleaned_text"] = df["text"].apply(preprocess_text)
df["label_id"] = df["label"].map(LABEL_TO_ID)

print("\nLabel counts:")
print(df["label"].value_counts())


# ----------------------------
# Split FIRST
# ----------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["cleaned_text"],
    df["label_id"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)


# ----------------------------
# Tokenizer fitted on TRAIN only
# ----------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

y_train = np.array(y_train)
y_test = np.array(y_test)


# ----------------------------
# Class weights
# ----------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {int(cls): float(w) for cls, w in zip(np.unique(y_train), weights)}

print("\nClass weights:")
print(class_weights)


# ----------------------------
# Build model
# ----------------------------
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ----------------------------
# Train
# ----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=8,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)


# ----------------------------
# Evaluate
# ----------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("\nFinal Weighted LSTM Accuracy:", accuracy)

y_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_probs, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=LABEL_ORDER,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# ----------------------------
# Save confusion matrix
# ----------------------------
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Weighted LSTM")
plt.colorbar()

plt.xticks(np.arange(3), LABEL_ORDER)
plt.yticks(np.arange(3), LABEL_ORDER)

for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=300)
plt.close()

print("Saved confusion matrix:", CM_PATH)


# ----------------------------
# Save ROC
# ----------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

plt.figure(figsize=(7, 5))

for i, label in enumerate(LABEL_ORDER):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Weighted LSTM")
plt.legend()
plt.tight_layout()
plt.savefig(ROC_PATH, dpi=300)
plt.close()

print("Saved ROC curve:", ROC_PATH)


# ----------------------------
# Test sample predictions
# ----------------------------
test_messages = [
    "hello how are you",
    "you are stupid",
    "kill yourself",
    "nobody likes you",
    "you are annoying"
]

print("\nSample predictions before saving:")

for msg in test_messages:
    clean = preprocess_text(msg)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)[0]
    pred = int(np.argmax(probs))

    print("\nTEXT:", msg)
    print("CLEANED:", clean)
    print("SEQUENCE:", seq)
    print("PROBS:", probs)
    print("PRED:", ID_TO_LABEL[pred])


# ----------------------------
# Save model and tokenizer
# ----------------------------
model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(LABELS_PATH, "wb") as f:
    pickle.dump(ID_TO_LABEL, f)

print("\nSaved model:", MODEL_PATH)
print("Saved tokenizer:", TOKENIZER_PATH)
print("Saved labels:", LABELS_PATH)
