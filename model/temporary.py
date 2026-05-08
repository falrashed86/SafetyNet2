from tensorflow.keras.models import load_model

model = load_model("weighted_lstm_model.h5", compile=False)

model.save("weighted_lstm_model_fixed.h5")
