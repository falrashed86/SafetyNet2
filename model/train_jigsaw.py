import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Load dataset
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "jigsaw.csv"

df = pd.read_csv(DATA_PATH)

# ----------------------------
# Convert labels
# ----------------------------
def map_label(row):
    if row["severe_toxic"] == 1 or row["threat"] == 1 or row["insult"] == 1 or row["obscene"] == 1 or row["identity_hate"] == 1:
        return "HIGH"
    elif row["toxic"] == 1:
        return "MEDIUM"
    else:
        return "LOW"

df["label"] = df.apply(map_label, axis=1)

# Keep only needed columns
df = df[["comment_text", "label"]]
df.columns = ["text", "label"]

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
# Vectorizer
# ----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2),
    max_features=10000,
    min_df=5
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test_vec)

print("\n=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
