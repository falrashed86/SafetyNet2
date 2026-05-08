from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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


df_jigsaw["label"] = df_jigsaw.apply(map_label, axis=1)
df_jigsaw = df_jigsaw[["comment_text", "label"]]
df_jigsaw.columns = ["text", "label"]


# ----------------------------
# Balance Jigsaw
# ----------------------------
df_low = df_jigsaw[df_jigsaw["label"] == "LOW"]
df_medium = df_jigsaw[df_jigsaw["label"] == "MEDIUM"]
df_high = df_jigsaw[df_jigsaw["label"] == "HIGH"]

# Use the size of HIGH as target so classes are closer
target_size = min(len(df_high), len(df_medium), 10000)

df_low_bal = df_low.sample(n=target_size, random_state=42)
df_medium_bal = df_medium.sample(n=target_size, random_state=42, replace=len(df_medium) < target_size)
df_high_bal = df_high.sample(n=target_size, random_state=42)

df_jigsaw_balanced = pd.concat([df_low_bal, df_medium_bal, df_high_bal], ignore_index=True)
df_jigsaw_balanced = df_jigsaw_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


# ----------------------------
# Load custom dataset
# ----------------------------
df_custom = pd.read_csv(CUSTOM_PATH, encoding="utf-8")
df_custom.columns = [str(c).strip().lower() for c in df_custom.columns]
df_custom = df_custom.dropna(subset=["text", "label"])
df_custom["text"] = df_custom["text"].astype(str)
df_custom["label"] = df_custom["label"].astype(str).str.upper().str.strip()

print("Balanced Jigsaw label counts:")
print(df_jigsaw_balanced["label"].value_counts())

print("\nCustom label counts:")
print(df_custom["label"].value_counts())


# ----------------------------
# Train on balanced Jigsaw
# ----------------------------
X_train = df_jigsaw_balanced["text"]
y_train = df_jigsaw_balanced["label"]

# Test on custom dataset
X_test = df_custom["text"]
y_test = df_custom["label"]


# ----------------------------
# Custom stop words
# ----------------------------
custom_stop_words = set(ENGLISH_STOP_WORDS) - {
    "you", "your", "yours", "not", "no", "never"
}

arabic_stop_words = {
    "في", "من", "على", "و", "الى", "إلى", "عن", "مع", "هذا", "هذه"
}

all_stop_words = list(custom_stop_words.union(arabic_stop_words))


# ----------------------------
# Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=all_stop_words,
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ----------------------------
# Train model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# ----------------------------
# Predict on custom dataset
# ----------------------------
y_pred = model.predict(X_test_vec)


# ----------------------------
# Evaluate on custom dataset
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Train on BALANCED Jigsaw, Test on Custom ===")
print("Accuracy:", accuracy)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
