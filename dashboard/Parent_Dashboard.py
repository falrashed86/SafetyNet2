import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import time
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from database.db import init_db, get_recent_messages, get_recent_high, get_risk_counts


st.set_page_config(
    page_title="SafetyNet Parent Dashboard",
    page_icon="👨‍👩‍👧",
    layout="wide"
)

init_db()

LOGO_PATH = BASE_DIR / "dashboard" / "assets" / "logo.png"

if "last_alert_id" not in st.session_state:
    st.session_state.last_alert_id = None

if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=150)

st.title("‍‍ SafetyNet Parent Dashboard")
st.write("This dashboard displays messages analysed using the final weighted LSTM model.")

counts = get_risk_counts()

col1, col2, col3 = st.columns(3)
col1.metric("LOW", counts["LOW"])
col2.metric("MEDIUM", counts["MEDIUM"])
col3.metric("HIGH", counts["HIGH"])

st.divider()

# ----------------------------
# Temporary high-risk notification + sound
# ----------------------------
high_alerts = get_recent_high(limit=1)

alert_placeholder = st.empty()

if len(high_alerts) > 0:
    latest_high = high_alerts[0]
    latest_high_id = latest_high[0]

    if st.session_state.last_alert_id != latest_high_id:
        st.session_state.last_alert_id = latest_high_id

        # Loud browser sound using Web Audio API
        components.html(
            """
            <script>
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

            function beep(freq, duration, delay) {
                const oscillator = audioCtx.createOscillator();
                const gainNode = audioCtx.createGain();

                oscillator.connect(gainNode);
                gainNode.connect(audioCtx.destination);

                oscillator.type = "square";
                oscillator.frequency.value = freq;

                gainNode.gain.setValueAtTime(0.9, audioCtx.currentTime + delay);
                gainNode.gain.exponentialRampToValueAtTime(
                    0.001,
                    audioCtx.currentTime + delay + duration
                );

                oscillator.start(audioCtx.currentTime + delay);
                oscillator.stop(audioCtx.currentTime + delay + duration);
            }

            beep(950, 0.45, 0.0);
            beep(1150, 0.45, 0.55);
            beep(950, 0.45, 1.10);
            beep(1150, 0.45, 1.65);
            beep(950, 0.45, 2.20);
            </script>
            """,
            height=0
        )

        alert_placeholder.error("🚨 New HIGH-risk message detected!")

        time.sleep(4)
        alert_placeholder.empty()

# ----------------------------
# Bar chart
# ----------------------------
st.subheader("Risk Level Distribution")

chart_df = pd.DataFrame({
    "Risk Level": ["LOW", "MEDIUM", "HIGH"],
    "Count": [counts["LOW"], counts["MEDIUM"], counts["HIGH"]]
})

st.bar_chart(chart_df.set_index("Risk Level"))

st.divider()

tab1, tab2 = st.tabs(["Recent Messages", "High Risk Alerts"])

columns = [
    "ID",
    "Message",
    "Cleaned Text",
    "Risk",
    "Confidence",
    "LOW Probability",
    "MEDIUM Probability",
    "HIGH Probability",
    "Model",
    "Time"
]

with tab1:
    rows = get_recent_messages(limit=50)

    if len(rows) == 0:
        st.info("No messages yet.")
    else:
        df = pd.DataFrame(rows, columns=columns)

        numeric_cols = [
            "Confidence",
            "LOW Probability",
            "MEDIUM Probability",
            "HIGH Probability"
        ]

        for col in numeric_cols:
            df[col] = df[col].round(3)

        show_text = st.checkbox("Show message text", value=False)

        if not show_text:
            df["Message"] = "Hidden for privacy"
            df["Cleaned Text"] = "Hidden for privacy"

        st.dataframe(df, use_container_width=True)

with tab2:
    rows = get_recent_high(limit=30)

    if len(rows) == 0:
        st.info("No high-risk alerts.")
    else:
        df_high = pd.DataFrame(rows, columns=columns)

        numeric_cols = [
            "Confidence",
            "LOW Probability",
            "MEDIUM Probability",
            "HIGH Probability"
        ]

        for col in numeric_cols:
            df_high[col] = df_high[col].round(3)

        show_high_text = st.checkbox("Show high-risk message text", value=False)

        if not show_high_text:
            df_high["Message"] = "Hidden for privacy"
            df_high["Cleaned Text"] = "Hidden for privacy"

        st.dataframe(df_high, use_container_width=True)
