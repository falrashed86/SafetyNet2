import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_autorefresh import st_autorefresh

#from database.db import init_db, get_counts, get_recent_high
from database.db import init_db, get_risk_counts, get_recent_high

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="SafetyNet – Parent Dashboard", layout="wide")

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
    .block-container{
        padding-top:1rem;
    }

    .sn-header{
        display:flex;
        align-items:center;
        gap:16px;
        background:linear-gradient(90deg,#e9f3ff,#f3ecff);
        padding:16px;
        border-radius:14px;
        border:1px solid #e2e2e2;
        margin-bottom: 10px;
    }

    .sn-title{
        font-size:24px;
        font-weight:700;
    }

    .sn-sub{
        font-size:14px;
        color:#666;
    }

    .small-note{
        font-size:13px;
        color:#666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
c1, c2 = st.columns([1.2, 5.8], vertical_alignment="center")

with c1:
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=140)
    else:
        st.warning(f"Logo not found: {LOGO_PATH}")

with c2:
    st.markdown(
        """
        <div class="sn-header">
            <div>
                <div class="sn-title">SafetyNet – Parent Dashboard</div>
                <div class="sn-sub">AI-powered child safety monitoring system</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Auto refresh
# ----------------------------
st_autorefresh(interval=1500, key="sn_parent_refresh")

# ----------------------------
# Initialize database
# ----------------------------
init_db()

# ----------------------------
# Sound alert
# ----------------------------
def play_beep():
    components.html(
        """
        <script>
        const audio = new Audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg");
        audio.volume = 0.9;

        let times = 0;

        function playAgain() {
            audio.currentTime = 0;
            audio.play().catch(e => console.log(e));
            times++;
            if (times < 4) {
                setTimeout(playAgain, 700);
            }
        }

        playAgain();
        </script>
        """,
        height=0,
    )

if "sound_enabled" not in st.session_state:
    st.session_state["sound_enabled"] = False

s1, s2 = st.columns([1.2, 4.8], vertical_alignment="center")

with s1:
    if st.button("🔊 Enable sound"):
        st.session_state["sound_enabled"] = True
        st.success("Sound enabled")

with s2:
    st.caption("Click once so the browser allows alert sounds.")

# ----------------------------
# Alert logic
# ----------------------------
counts = get_risk_counts()

low_now = counts["LOW"]
medium_now = counts["MEDIUM"]
high_now = counts["HIGH"]

total_now = low_now + medium_now + high_now
now = time.time()

if "prev_high" not in st.session_state:
    st.session_state["prev_high"] = high_now

if "alert_until" not in st.session_state:
    st.session_state["alert_until"] = 0.0

if high_now > st.session_state["prev_high"]:
    st.session_state["prev_high"] = high_now
    st.session_state["alert_until"] = now + 8
    st.toast("New high-risk alert received", icon="⚠️")

    if st.session_state["sound_enabled"]:
        play_beep()

alert_box = st.empty()

if now < st.session_state["alert_until"]:
    with alert_box.container():
        st.error("⚠️ New HIGH-RISK alert detected")
        if st.button("Dismiss alert"):
            st.session_state["alert_until"] = 0.0
            st.rerun()

# ----------------------------
# Metrics
# ----------------------------
m1, m2 = st.columns(2)
m1.metric("Total messages analysed", total_now)
m2.metric("High-risk alerts", high_now)

st.divider()

# ----------------------------
# Recent analysis
# ----------------------------
st.subheader("Recent Messages Analysis")
st.caption("Shows the final decision, your trained model output, and BERT output.")

show_text = st.checkbox("Show message text (privacy off)", value=False)

rows = get_recent_high(limit=30)

if not rows:
    st.info("No messages yet.")
else:
    df = pd.DataFrame(rows, columns=[
        "id",
        "text",
        "mode",
        "risk",
        "trained_prediction",
        "trained_confidence",
        "bert_label",
        "bert_stars",
        "bert_confidence",
        "bert_risk",
        "created_at"
    ])

    df = df.fillna("")

    def fmt_conf(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return ""

    if show_text:
        message_col = df["text"]
    else:
        message_col = ["[Hidden for privacy]"] * len(df)

    df_display = pd.DataFrame({
        "Message": message_col,
        "Final Risk": df["risk"],
        "Mode": df["mode"],
        "Your Model Prediction": df["trained_prediction"],
        "Your Model Confidence": df["trained_confidence"].apply(fmt_conf),
        "BERT Label": df["bert_label"],
        "BERT Risk": df["bert_risk"],
        "BERT Confidence": df["bert_confidence"].apply(fmt_conf),
        "Time": df["created_at"]
    })

    st.dataframe(df_display, use_container_width=True)

    # ----------------------------
    # Trend tracking
    # ----------------------------
    st.divider()
    st.subheader("Trend Overview")
    st.caption("Simple trend tracking based on stored model predictions.")

    counts = df["risk"].value_counts()

    trend_df = pd.DataFrame({
        "Risk": counts.index,
        "Count": counts.values
    }).set_index("Risk")

    st.bar_chart(trend_df)

st.markdown(
    """
    <div class='small-note'>
    Final decision uses your trained model. BERT is shown for comparison.
    Trend tracking uses stored predictions over time to summarize behavior.
    </div>
    """,
    unsafe_allow_html=True
)
