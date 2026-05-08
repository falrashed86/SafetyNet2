import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from model.text_analyzer import analyze_text
from database.db import init_db, insert_message


st.set_page_config(
    page_title="SafetyNet – Child App",
    page_icon="🛡️",
    layout="centered"
)

init_db()

LOGO_PATH = ROOT_DIR / "dashboard" / "assets" / "logo.png"

if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=140)

st.title("🛡️ SafetyNet Child App")
st.write(
    "This page simulates a child sending a message. "
    "The message is analysed using the final weighted LSTM model."
)

message = st.text_area("Type a message:", height=150)

send_button = st.button("Send Message")

if send_button:
    if message.strip() == "":
        st.warning("Please type a message first.")
    else:
        result = analyze_text(message)

        insert_message(
            text=result["text"],
            cleaned_text=result["cleaned_text"],
            risk=result["risk"],
            confidence=result["confidence"],
            low_prob=result["low_prob"],
            medium_prob=result["medium_prob"],
            high_prob=result["high_prob"],
            mode=result["mode"]
        )

        st.subheader("Analysis Result")

        if result["risk"].upper() == "HIGH":
            st.error("🚨 HIGH Risk Message Detected")
        elif result["risk"].upper() == "MEDIUM":
            st.warning("⚠️ MEDIUM Risk Message Detected")
        else:
            st.success("✅ LOW Risk Message")

        st.write("**Model:**", result["mode"])
        st.write("**Risk Level:**", result["risk"])
        st.write("**Confidence:**", round(result["confidence"], 3))

        st.write("### Class Probabilities")
        st.write("LOW:", round(result["low_prob"], 3))
        st.write("MEDIUM:", round(result["medium_prob"], 3))
        st.write("HIGH:", round(result["high_prob"], 3))

        st.success("Message analysed and saved.")
