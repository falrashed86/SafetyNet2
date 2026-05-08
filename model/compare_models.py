from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# choose one:
DATA_PATH = BASE_DIR / "data" / "jigsaw.csv"      # for high-accuracy comparison
# DATA_PATH = BASE_DIR / "data" / "messages.csv"  # for your custom dataset


# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------
# If Jigsaw, convert labels
# ----------------------------
if "comment_text" in df.columns:
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

else:
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[["text", "label"]]

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str).str.upper().str.strip()

print("Dataset loaded successfully.")
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
# Stop words
# ----------------------------
custom_stop_words = set(ENGLISH_STOP_WORDS) - {
    "you", "your", "yours", "not", "no", "never"
}

arabic_stop_words = {
    "في", "من", "على", "و", "الى", "إلى", "عن", "مع", "هذا", "هذه"
}

all_stop_words = list(custom_stop_words.union(arabic_stop_words))


# ----------------------------
# Vectorizer
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
# Models to compare
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=2,
        class_weight="balanced"
    ),
    "Naive Bayes": MultinomialNB(alpha=0.5),
    "Bernoulli NB": BernoulliNB(),
    "Linear SVM": LinearSVC(),
    "SGD Classifier": SGDClassifier(
        loss="hinge",
        max_iter=1000,
        random_state=42
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
}


# ----------------------------
# Train + evaluate all
# ----------------------------
results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {name}")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))

    print(f"\n=== {name} Accuracy ===")
    print(acc)

    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred))

    print(f"\n=== {name} Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))


# ----------------------------
# Final summary
# ----------------------------
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

results = sorted(results, key=lambda x: x[1], reverse=True)
for name, acc in results:
    print(f"{name}: {acc:.4f}")
