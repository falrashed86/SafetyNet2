from model.lstm_predictor import predict_risk


def analyze_text(text):
    return predict_risk(text)
