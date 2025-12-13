# Analyzer_app.py

import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data_processing import load_data
from predict import predict_for_product
from chatbot_streamlit import get_product_map as get_chatbot_map, chatbot_response

# --- 1. CONFIGURATION ---

st.set_page_config(
    page_title="Demand Forecasting AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BASE DIRECTORY (IMPORTANT FIX) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- 2. BACKGROUND EFFECT ---

def add_3d_background():
    html_code = """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.net.min.js"></script>
    <style>
        #vanta-canvas {
            position: fixed;
            left: 0;
            top: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            opacity: 0.5;
        }
    </style>
    <div id="vanta-canvas"></div>
    <script>
        VANTA.NET({
            el: "#vanta-canvas",
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            scale: 1.0,
            scaleMobile: 1.0,
            color: 0xa97458,
            backgroundColor: 0x1a1a1a,
            points: 12.0,
            maxDistance: 22.0,
            spacing: 18.0
        })
    </script>
    """
    components.html(html_code, height=0, width=0)

add_3d_background()


# --- 3. DATA HELPERS ---

@st.cache_data
def get_product_map():
    df = load_data()
    product_df = df[['product_id', 'product_name']].drop_duplicates()
    product_list = sorted(product_df['product_name'].unique())
    name_to_id = product_df.set_index('product_name')['product_id'].to_dict()
    return product_list, name_to_id


@st.cache_data
def get_historical_data(product_id):
    df = load_data()
    df = df[df['product_id'] == product_id].sort_values('date').tail(90)
    df['Type'] = 'Historical'
    return df


# --- 4. CSS STYLING ---

st.markdown(
    """
    <style>
    .stApp { background: transparent !important; }

    section[data-testid="stSidebar"] {
        background-color: rgba(20,20,20,0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    h1, h2, h3, h4, h5, p, label, span, div {
        color: #e0e0e0 !important;
        font-family: 'Segoe UI', Tahoma;
    }

    .stButton > button {
        background: linear-gradient(90deg, #A87456, #8c5e42);
        color: white;
        border-radius: 6px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- 5. HEADER ---

st.markdown("<h1>Predictive Analytics for Inventory Management</h1>", unsafe_allow_html=True)
st.markdown("### Demand Forecasting")
st.write("---")

product_list, name_to_id = get_product_map()


# --- 6. SIDEBAR ---

with st.sidebar:

    # ✅ LOGO FIX (ABSOLUTE PATH)
    logo_path = os.path.join(BASE_DIR, "final_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.warning("⚠️ final_logo.png not found")

    st.markdown("## ⚙️ Forecast Settings")

    selected_product_name = st.selectbox(
        "Select Product",
        product_list,
        index=0
    )

    selected_product_id = name_to_id.get(selected_product_name)
    days_ahead = st.slider("Forecast Horizon (Days)", 7, 90, 30, 7)
    run_forecast = st.button("🚀 Generate Forecast")


# --- 7. MAIN LOGIC ---

if run_forecast or st.session_state.get("initial_run", True):
    st.session_state["initial_run"] = False

    if selected_product_id:
        with st.spinner("Running AI Forecast..."):
            forecast = predict_for_product(selected_product_id, days_ahead)

        forecast_df = pd.DataFrame(forecast)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df.rename(columns={'predicted_demand': 'demand'}, inplace=True)
        forecast_df['Type'] = 'Forecast'

        history_df = get_historical_data(selected_product_id)

        combined_df = pd.concat([
            history_df[['date', 'demand', 'price', 'Type']],
            forecast_df[['date', 'demand', 'price', 'Type']]
        ])

        st.subheader(f"📊 Results: {selected_product_name}")

        total = forecast_df['demand'].sum()
        avg = forecast_df['demand'].mean()

        col1, col2 = st.columns([1, 4])

        with col1:
            st.metric("Total Forecast", f"{total:,.0f}")
            st.metric("Daily Avg", f"{avg:.1f}")

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df['date'], y=history_df['demand'],
                name="Historical", mode="lines",
                line=dict(color="#d4a373")
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['demand'],
                name="Forecast", mode="lines+markers",
                line=dict(color="#00ffd0", width=3)
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)


# --- 8. CHATBOT ---

st.markdown("---")
st.header("🤖 Chatbot Assistant")

if "chatbot_map" not in st.session_state:
    st.session_state.chatbot_map = get_chatbot_map()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask about demand, products, trends...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    reply = chatbot_response(user_input, st.session_state.chatbot_map)
    st.session_state.chat_history.append(("bot", reply))

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(f"<span style='color:#d4a373'>{msg}</span>", unsafe_allow_html=True)
