from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "messages.csv"
MODEL_PATH = BASE_DIR / "model" / "trained_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"


# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df.columns = [str(c).strip().lower() for c in df.columns]

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
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
# Custom stop words
# Keep important words that affect meaning
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
    max_features=5000,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ----------------------------
# Train model
# ----------------------------
model = LogisticRegression(
    max_iter=1000,
    C=2,
    class_weight="balanced"
)
model.fit(X_train_vec, y_train)


# ----------------------------
# Predict
# ----------------------------
y_pred = model.predict(X_test_vec)


# ----------------------------
# Evaluate
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Accuracy ===")
print(accuracy)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))


# ----------------------------
# Save model + vectorizer
# ----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved to:", MODEL_PATH)
print("Vectorizer saved to:", VECTORIZER_PATH)
