from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
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
OUT_DIR = BASE_DIR / "outputs" / "matrices_roc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_PATH = DATA_DIR / "messages.csv"
JIGSAW_PATH = DATA_DIR / "jigsaw.csv"

custom_stop_words = set(ENGLISH_STOP_WORDS)
for word in ["not", "no", "never"]:
    custom_stop_words.discard(word)


def clean_filename(name):
    return name.replace(" ", "_").replace("/", "_").replace("-", "_")


def load_custom():
    df = pd.read_csv(CUSTOM_PATH, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    return df[["text", "label"]]


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


def save_confusion_matrix(y_test, y_pred, labels_text, dataset_name, model_name):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name} ({dataset_name})")
    plt.colorbar()

    tick_marks = np.arange(len(labels_text))
    plt.xticks(tick_marks, labels_text)
    plt.yticks(tick_marks, labels_text)

    for i in range(len(labels_text)):
        for j in range(len(labels_text)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    filename = OUT_DIR / f"cm_{clean_filename(dataset_name)}_{clean_filename(model_name)}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Saved:", filename)


def save_roc_curve(y_test, y_score, labels_encoded, labels_text, dataset_name, model_name):
    try:
        y_test_bin = label_binarize(y_test, classes=labels_encoded)

        plt.figure(figsize=(7, 5))

        for i, label in enumerate(labels_text):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} AUC = {roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name} ({dataset_name})")
        plt.legend()
        plt.tight_layout()

        filename = OUT_DIR / f"roc_{clean_filename(dataset_name)}_{clean_filename(model_name)}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print("Saved:", filename)

    except Exception as e:
        print("ROC skipped for", model_name, dataset_name, "because:", e)


def softmax(scores):
    scores = np.array(scores)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def run_ml_models(df, dataset_name):
    X = df["text"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    labels_encoded = np.arange(len(encoder.classes_))
    labels_text = encoder.classes_

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
        print(f"Training {model_name} on {dataset_name}")

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
            y_score = softmax(pipe.decision_function(X_test))
        else:
            y_score = np.zeros((len(y_test), len(labels_encoded)))

        save_confusion_matrix(y_test, y_pred, labels_text, dataset_name, model_name)
        save_roc_curve(y_test, y_score, labels_encoded, labels_text, dataset_name, model_name)


def prepare_dl(df):
    X = df["text"].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    labels_encoded = np.arange(len(encoder.classes_))
    labels_text = encoder.classes_

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

    return X_train, X_test, y_train, y_test, labels_encoded, labels_text, class_weight_dict


def run_cnn(df, dataset_name):
    print(f"Training CNN on {dataset_name}")

    X_train, X_test, y_train, y_test, labels_encoded, labels_text, class_weight_dict = prepare_dl(df)

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=100),
        Conv1D(64, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(labels_encoded), activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=3, batch_size=32, class_weight=class_weight_dict, verbose=1)

    y_score = model.predict(X_test)
    y_pred = np.argmax(y_score, axis=1)

    save_confusion_matrix(y_test, y_pred, labels_text, dataset_name, "CNN")
    save_roc_curve(y_test, y_score, labels_encoded, labels_text, dataset_name, "CNN")


def run_lstm(df, dataset_name):
    print(f"Training LSTM on {dataset_name}")

    X_train, X_test, y_train, y_test, labels_encoded, labels_text, class_weight_dict = prepare_dl(df)

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=100),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(labels_encoded), activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=3, batch_size=32, class_weight=class_weight_dict, verbose=1)

    y_score = model.predict(X_test)
    y_pred = np.argmax(y_score, axis=1)

    save_confusion_matrix(y_test, y_pred, labels_text, dataset_name, "LSTM")
    save_roc_curve(y_test, y_score, labels_encoded, labels_text, dataset_name, "LSTM")


def run_dataset(df, dataset_name):
    print("\n==============================")
    print(dataset_name)
    print("==============================")
    run_ml_models(df, dataset_name)
    run_cnn(df, dataset_name)
    run_lstm(df, dataset_name)


custom_df = load_custom()
jigsaw_df = load_jigsaw()
combined_df = pd.concat([custom_df, jigsaw_df], ignore_index=True)

run_dataset(custom_df, "Custom Dataset")
run_dataset(jigsaw_df, "Jigsaw Dataset")
run_dataset(combined_df, "Combined Dataset")

print("\nAll confusion matrices and ROC curves saved in:")
print(OUT_DIR)
