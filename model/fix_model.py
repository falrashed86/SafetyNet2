from tensorflow.keras.models import load_model

model = load_model("model/weighted_lstm_model.h5", compile=False)
model.save("model/weighted_lstm_model.keras")
print("Saved fixed model")
