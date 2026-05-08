from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CUSTOM_PATH = DATA_DIR / "messages.csv"
JIGSAW_PATH = DATA_DIR / "jigsaw.csv"


# ----------------------------
# Load custom dataset
# ----------------------------
def load_custom():
    df = pd.read_csv(CUSTOM_PATH, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    return df[["text", "label"]]


# ----------------------------
# Load Jigsaw dataset
# ----------------------------
def load_jigsaw():
    df = pd.read_csv(JIGSAW_PATH)
    df = df.dropna(subset=["comment_text"])

    def map_label(row):
        if row["severe_toxic"] or row["threat"] or row["insult"] or row["obscene"] or row["identity_hate"]:
            return "HIGH"
        elif row["toxic"]:
            return "MEDIUM"
        else:
            return "LOW"

    df["label"] = df.apply(map_label, axis=1)
    df = df[["comment_text", "label"]]
    df.columns = ["text", "label"]
    return df


# ----------------------------
# Metrics
# ----------------------------
def calculate_metrics(y_test, y_pred, y_prob, labels):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        y_test_bin = label_binarize(y_test, classes=labels)
        auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    except:
        auc = np.nan

    return acc, precision, recall, f1, auc


# ----------------------------
# ML Models
# ----------------------------
def run_ml(df, dataset_name):
    results = []

    X = df["text"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    labels = np.arange(len(encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # IMPORTANT: Avoid SVM/KNN for Jigsaw
    if dataset_name == "Jigsaw Dataset":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(kernel="linear", probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
        }

    for name, model in models.items():
        print(f"Running {name} on {dataset_name}...")

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000)),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)
        else:
            y_prob = np.zeros((len(y_test), len(labels)))

        acc, p, r, f1, auc = calculate_metrics(y_test, y_pred, y_prob, labels)

        results.append({
            "Dataset": dataset_name,
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4) if not np.isnan(auc) else "N/A"
        })

    return results


# ----------------------------
# Deep Learning
# ----------------------------
def prepare_dl(df):
    X = df["text"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    labels = np.arange(len(encoder.classes_))

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100)

    return train_test_split(X_pad, y, test_size=0.2, stratify=y), labels


def run_lstm(df, dataset):
    print(f"Running LSTM on {dataset}...")

    (X_train, X_test, y_train, y_test), labels = prepare_dl(df)

    model = Sequential([
        Embedding(10000, 128),
        LSTM(64),
        Dense(3, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=3, batch_size=32)

    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    acc, p, r, f1, auc = calculate_metrics(y_test, y_pred, y_prob, labels)

    return {
        "Dataset": dataset,
        "Model": "LSTM",
        "Accuracy": round(acc, 4),
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f1, 4),
        "AUC": round(auc, 4)
    }


def run_cnn(df, dataset):
    print(f"Running CNN on {dataset}...")

    (X_train, X_test, y_train, y_test), labels = prepare_dl(df)

    model = Sequential([
        Embedding(10000, 128),
        Conv1D(64, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(3, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=3, batch_size=32)

    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    acc, p, r, f1, auc = calculate_metrics(y_test, y_pred, y_prob, labels)

    return {
        "Dataset": dataset,
        "Model": "CNN",
        "Accuracy": round(acc, 4),
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f1, 4),
        "AUC": round(auc, 4)
    }


# ----------------------------
# RUN ALL
# ----------------------------
custom = load_custom()
jigsaw = load_jigsaw()
combined = pd.concat([custom, jigsaw])

results = []
results += run_ml(custom, "Custom Dataset")
results += run_ml(jigsaw, "Jigsaw Dataset")
results += run_ml(combined, "Combined Dataset")

results.append(run_cnn(jigsaw, "Jigsaw Dataset"))
results.append(run_lstm(jigsaw, "Jigsaw Dataset"))

df = pd.DataFrame(results)

print("\nFINAL TABLE:\n", df)

df.to_csv(OUT_DIR / "final_results.csv", index=False)

print("\nSaved to outputs/final_results.csv")
