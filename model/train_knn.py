from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "messages.csv"
MODEL_PATH = BASE_DIR / "model" / "knn_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "knn_vectorizer.pkl"


# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df.columns = [str(c).strip().lower() for c in df.columns]

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str).str.upper().str.strip()

print("Dataset loaded")
print(df.head())


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
    ngram_range=(1,2),      # 🔥 BIG improvement
    max_features=5000,      # limit noise
    min_df=2                # ignore rare words
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ----------------------------
# Train model
# ----------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_vec, y_train)


# ----------------------------
# Predict
# ----------------------------
y_pred = model.predict(X_test_vec)


# ----------------------------
# Evaluate
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n=== KNN Accuracy ===")
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

print("\nKNN model saved to:", MODEL_PATH)
print("KNN vectorizer saved to:", VECTORIZER_PATH)
