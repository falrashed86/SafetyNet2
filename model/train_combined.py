from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
JIGSAW_PATH = BASE_DIR / "data" / "jigsaw.csv"
CUSTOM_PATH = BASE_DIR / "data" / "messages.csv"


# ----------------------------
# Load Jigsaw
# ----------------------------
df_jigsaw = pd.read_csv(JIGSAW_PATH)
df_jigsaw = df_jigsaw.dropna(subset=["comment_text"])
df_jigsaw["comment_text"] = df_jigsaw["comment_text"].astype(str)


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
df_combined = pd.concat([df_jigsaw, df_custom], ignore_index=True)

print("Combined label counts:")
print(df_combined["label"].value_counts())


# ----------------------------
# Train/test split (IMPORTANT)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_combined["text"],
    df_combined["label"],
    test_size=0.2,
    random_state=42,
    stratify=df_combined["label"]
)


# ----------------------------
# Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ----------------------------
# Train model
# ----------------------------
model = LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(X_train_vec, y_train)


# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test_vec)

print("\n=== Combined Training Result ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
