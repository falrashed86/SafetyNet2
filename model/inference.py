from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "trained_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"


def load_trained_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_risk(text: str):
    model, vectorizer = load_trained_model()

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    # probability support if available
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        class_to_prob = dict(zip(classes, probs))
        confidence = float(class_to_prob.get(prediction, 0.0))

    return {
        "prediction": prediction,
        "confidence": confidence
    }
