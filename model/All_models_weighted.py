from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CUSTOM_PATH = DATA_DIR / "messages.csv"
JIGSAW_PATH = DATA_DIR / "jigsaw.csv"


# ----------------------------
# Custom stop words for ML
# ----------------------------
custom_stop_words = set(ENGLISH_STOP_WORDS)
for important_word in ["not", "no", "never"]:
    custom_stop_words.discard(important_word)


# ----------------------------
# Load custom dataset
# ----------------------------
def load_custom():
    df = pd.read_csv(CUSTOM_PATH, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
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
    df["comment_text"] = df["comment_text"].astype(str)

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
# Metrics
# ----------------------------
def calculate_metrics(y_test, y_pred, y_score, labels):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        y_test_bin = label_binarize(y_test, classes=labels)
        auc = roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    return acc, precision, recall, f1, auc


def softmax(scores):
    scores = np.array(scores)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# ----------------------------
# Machine learning models
# ----------------------------
def run_ml_models(df, dataset_name):
    results = []

    X = df["text"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    labels = np.arange(len(encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "KNN": Pipeline([
            ("svd", TruncatedSVD(n_components=100, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    for model_name, model in models.items():
        print(f"\nRunning {model_name} on {dataset_name}...")

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words=list(custom_stop_words),
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2
            )),
            ("model", model)
        ])

        if model_name == "Naive Bayes":
            pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
        else:
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_score = pipe.predict_proba(X_test)
        elif hasattr(pipe.named_steps["model"], "decision_function"):
            decision_scores = pipe.decision_function(X_test)
            y_score = softmax(decision_scores)
        else:
            y_score = np.zeros((len(y_test), len(labels)))

        acc, precision, recall, f1, auc = calculate_metrics(y_test, y_pred, y_score, labels)

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "AUC": round(auc, 4) if not np.isnan(auc) else "N/A"
        })

    return results


# ----------------------------
# Deep learning preparation
# ----------------------------
def prepare_dl(df):
    X = df["text"].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    labels = np.arange(len(encoder.classes_))

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    return X_train, X_test, y_train, y_test, labels, class_weight_dict


# ----------------------------
# CNN
# ----------------------------
def run_cnn(df, dataset_name):
    print(f"\nRunning CNN on {dataset_name}...")

    X_train, X_test, y_train, y_test, labels, class_weight_dict = prepare_dl(df)

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=100),
        Conv1D(64, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(labels), activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )

    y_score = model.predict(X_test)
    y_pred = np.argmax(y_score, axis=1)

    acc, precision, recall, f1, auc = calculate_metrics(y_test, y_pred, y_score, labels)

    return {
        "Dataset": dataset_name,
        "Model": "CNN",
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else "N/A"
    }


# ----------------------------
# LSTM
# ----------------------------
def run_lstm(df, dataset_name):
    print(f"\nRunning LSTM on {dataset_name}...")

    X_train, X_test, y_train, y_test, labels, class_weight_dict = prepare_dl(df)

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=100),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(labels), activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )

    y_score = model.predict(X_test)
    y_pred = np.argmax(y_score, axis=1)

    acc, precision, recall, f1, auc = calculate_metrics(y_test, y_pred, y_score, labels)

    return {
        "Dataset": dataset_name,
        "Model": "LSTM",
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else "N/A"
    }


# ----------------------------
# Run one dataset
# ----------------------------
def run_dataset(df, dataset_name):
    print("\n==============================")
    print(dataset_name)
    print("==============================")
    print(df["label"].value_counts())

    results = []

    results.extend(run_ml_models(df, dataset_name))
    results.append(run_cnn(df, dataset_name))
    results.append(run_lstm(df, dataset_name))

    return results


# ----------------------------
# Main
# ----------------------------
custom_df = load_custom()
jigsaw_df = load_jigsaw()
combined_df = pd.concat([custom_df, jigsaw_df], ignore_index=True)

all_results = []
all_results.extend(run_dataset(custom_df, "Custom Dataset"))
all_results.extend(run_dataset(jigsaw_df, "Jigsaw Dataset"))
all_results.extend(run_dataset(combined_df, "Combined Dataset"))

results_df = pd.DataFrame(all_results)

print("\n\nFINAL RESULTS:")
print(results_df)

results_df.to_csv(OUT_DIR / "all_model_results_weighted.csv", index=False)

for dataset_name in results_df["Dataset"].unique():
    table = results_df[results_df["Dataset"] == dataset_name]
    safe_name = dataset_name.lower().replace(" ", "_")
    table.to_csv(OUT_DIR / f"{safe_name}_weighted_results.csv", index=False)

print("\nSaved files in:")
print(OUT_DIR)
