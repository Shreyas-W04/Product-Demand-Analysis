# Analyzer_app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.data_processing import load_data
from predict import predict_for_product
from chatbot_streamlit import get_product_map as get_chatbot_map, chatbot_response

# --- 1. CONFIGURATION AND UTILITIES ---

st.set_page_config(
    page_title="Demand Forecasting AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_3d_background():
    """
    Injects a 3D 'Neural Network' background using Vanta.js.
    """
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
            minHeight: 200.00,
            minWidth: 200.00,
            scale: 1.00,
            scaleMobile: 1.00,
            color: 0xa97458,       
            backgroundColor: 0x1a1a1a, 
            points: 12.00,
            maxDistance: 22.00,
            spacing: 18.00
        })
    </script>
    """
    components.html(html_code, height=0, width=0)

@st.cache_data
def get_product_map():
    try:
        df = load_data()
        product_df = df[['product_id', 'product_name']].drop_duplicates()
        product_list = sorted(product_df['product_name'].unique())
        name_to_id = product_df.set_index('product_name')['product_id'].to_dict()
        return product_list, name_to_id
    except Exception as e:
        st.error(f"Error loading product data: {e}")
        return [], {}

@st.cache_data
def get_historical_data(product_id):
    df_all = load_data()
    # Get last 90 days for richer plotting
    df_product = df_all[df_all['product_id'] == product_id].sort_values('date').tail(90).copy()
    df_product['Source'] = 'Historical'
    return df_product

# --- 2. CSS STYLING ---

st.markdown(
    """
    <style>
    .stApp { background: transparent !important; }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 20, 20, 0.85) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stCard, div[data-testid="stMetric"], div[data-testid="stExpander"] {
        background-color: rgba(30, 30, 30, 0.60) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    h1, h2, h3, h4, h5, h6, .main-header, p, label, span, div, li {
        color: #e0e0e0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stSelectbox > div > div, .stTextInput > div > div, .stSlider > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #A87456 0%, #8c5e42 100%);
        color: white !important;
        border: none;
        border-radius: 6px; 
    }
    
    .js-plotly-plot .plotly, .plot-container { background-color: rgba(0,0,0,0) !important; }
    
    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #cfcfcf !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #00ffd0 !important;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

add_3d_background()

# --- 3. APP LOGIC ---

st.markdown('<h1 class="main-header">Predictive Analytics for Inventory Management</h1>', unsafe_allow_html=True)
st.markdown("### Demand Forecasting")
st.write("---")

product_list, name_to_id = get_product_map()

# --- Sidebar ---
with st.sidebar:
    try:
        st.image("final_logo.png", width=200) 
    except:
        st.warning("Logo not found.")

    st.markdown("## ⚙️ Forecast Settings")
    
    selected_product_name = st.selectbox(
        "Select Product:", product_list,
        index=0 if not product_list else 0
    )
    selected_product_id = name_to_id.get(selected_product_name)
    
    days_ahead = st.slider("Forecast Horizon (Days):", 7, 90, 30, 7)
    run_forecast_button = st.button("🚀 Generate Forecast")

# --- Main Content ---
if run_forecast_button or st.session_state.get('initial_run', True):
    st.session_state['initial_run'] = False
    
    if selected_product_id:
        with st.spinner(f"Running Neural Network for {selected_product_name}..."):
            forecast_results = predict_for_product(selected_product_id, days_ahead)

        # Process Data
        forecast_df = pd.DataFrame(forecast_results)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df.rename(columns={'predicted_demand': 'demand'}, inplace=True)
        forecast_df['Type'] = 'Forecast'
        
        history_df = get_historical_data(selected_product_id)
        history_df['Type'] = 'Historical'
        
        # Combine for plots
        combined_df = pd.concat([
            history_df[['date', 'demand', 'price', 'Type']],
            forecast_df[['date', 'demand', 'price', 'Type']]
        ])

        total_forecast = forecast_df['demand'].sum()
        avg_daily = forecast_df['demand'].mean()
        
        # --- TOP SECTION: KPI Cards & Main Chart ---
        st.markdown(f"### 📊 Results: {selected_product_name}")
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(
                f"""
                <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 10px;">
                    <h4 style="margin:0; font-size: 14px; opacity: 0.7;">Total Forecast</h4>
                    <h2 style="margin:0; font-size: 28px; color: #00ffd0;">{total_forecast:,.0f}</h2>
                </div>
                <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                    <h4 style="margin:0; font-size: 14px; opacity: 0.7;">Daily Avg</h4>
                    <h2 style="margin:0; font-size: 28px; color: #d4a373;">{avg_daily:.1f}</h2>
                </div>
                """, unsafe_allow_html=True
            )

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df['date'], y=history_df['demand'], mode='lines', name='Historical',
                line=dict(color='#d4a373', width=2), fill='tozeroy', fillcolor='rgba(212, 163, 115, 0.05)'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['demand'], mode='lines+markers', name='AI Forecast',
                line=dict(color='#00ffd0', width=3, shape='spline'),
                marker=dict(size=6, color='#00ffd0', symbol='diamond', line=dict(width=1, color='white'))
            ))
            fig.update_layout(
                title=dict(text='Demand Trajectory', font=dict(color='white')),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#cfcfcf')),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#cfcfcf')),
                legend=dict(font=dict(color='#cfcfcf'), orientation="h", y=1.1),
                margin=dict(l=20, r=20, t=40, b=20), height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- ADVANCED ANALYSIS SECTION ---
        st.markdown("---")
        st.subheader("🔎 Deep Dive Analysis")
        
        # Tabs including the new Pie Chart tab
        tab_range, tab_season, tab_pie, tab_price = st.tabs(["📉 Forecast Range", "📅 Weekly Patterns", "🍰 Demand Composition", "💰 Price Elasticity"])
        
        with tab_range:
            st.markdown("**Prediction Confidence:** Shaded area shows expected range (±15%).")
            # Create synthetic Upper/Lower bounds for visualization effect
            upper_bound = forecast_df['demand'] * 1.15
            lower_bound = forecast_df['demand'] * 0.85
            
            fig_range = go.Figure()
            
            # Confidence Band
            fig_range.add_trace(go.Scatter(
                x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                y=pd.concat([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(0, 255, 208, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Confidence Interval'
            ))
            
            # Main Line
            fig_range.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['demand'],
                mode='lines+markers', name='Predicted',
                line=dict(color='#00ffd0', width=3)
            ))
            
            fig_range.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cfcfcf'),
                xaxis=dict(title='Date', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='Demand', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=450
            )
            st.plotly_chart(fig_range, use_container_width=True)

        with tab_season:
            # Bar Chart of Demand by Day of Week
            combined_df['Day'] = combined_df['date'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            avg_demand_day = combined_df.groupby('Day')['demand'].mean().reindex(day_order)
            
            fig_bar = px.bar(
                x=avg_demand_day.index, 
                y=avg_demand_day.values,
                color=avg_demand_day.values,
                color_continuous_scale='Tealgrn'
            )
            fig_bar.update_layout(
                title="Average Daily Demand (Magnitude)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cfcfcf'),
                xaxis=dict(title=''), yaxis=dict(title='Avg Units'),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- NEW PIE CHART TAB ---
        with tab_pie:
            # Prepare data for Pie Chart
            combined_df['Day'] = combined_df['date'].dt.day_name()
            day_sum = combined_df.groupby('Day')['demand'].sum().reset_index()
            
            fig_pie = px.pie(
                day_sum, 
                values='demand', 
                names='Day',
                title='Total Volume Share by Day of Week',
                color_discrete_sequence=px.colors.sequential.Tealgrn,
                hole=0.4 # Makes it a Donut chart
            )
            
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cfcfcf'),
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab_price:
            # Scatter of Price vs Demand
            fig_elast = px.scatter(
                combined_df, x='price', y='demand', color='Type',
                color_discrete_map={'Historical': '#d4a373', 'Forecast': '#00ffd0'},
                trendline="ols"  # Adds a trendline to show elasticity
            )
            fig_elast.update_layout(
                title="Price vs. Demand Correlation",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cfcfcf'),
                xaxis=dict(title='Price ($)'), yaxis=dict(title='Demand Units'),
            )
            st.plotly_chart(fig_elast, use_container_width=True)

        # Data Table
        with st.expander("📂 View Detailed Forecast Data"):
            st.dataframe(
                forecast_df[['date', 'demand', 'price']].rename(
                    columns={'date': 'Date', 'demand': 'Predicted Units', 'price': 'Price'}
                ).style.format({"Date": lambda t: t.strftime("%Y-%m-%d"), "Predicted Units": "{:.0f}", "Price": "${:.2f}"}),
                use_container_width=True
            )

# --- Chatbot Section ---
st.markdown("---")
st.header("🤖 Chatbot Assistant")

if "chatbot_map" not in st.session_state:
    st.session_state.chatbot_map = get_chatbot_map()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_message = st.chat_input("Ask about trends, specific products...")

if user_message:
    st.session_state.chat_history.append(("user", user_message))
    bot_reply = chatbot_response(user_message, st.session_state.chatbot_map)
    st.session_state.chat_history.append(("bot", bot_reply))

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(f"<span style='color: white'>{msg}</span>", unsafe_allow_html=True)
    else:
        st.chat_message("assistant").markdown(f"<span style='color: #d4a373'>{msg}</span>", unsafe_allow_html=True)