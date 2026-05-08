from tensorflow.keras.models import load_model

model = load_model("model/weighted_lstm_model.keras")

model.save("model/weighted_lstm_model.h5")

print("Model converted successfully.")
