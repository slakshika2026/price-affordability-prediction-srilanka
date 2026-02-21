import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="🇱🇰 Sri Lanka Food Price Predictor",
                   page_icon="🥗", layout="wide")

st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 5px !important;
    }
    
    .header-container {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 15px 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 6px 25px rgba(46, 204, 113, 0.3);
    }
    
    .header-container h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 2px 0;
    }
    
    .header-container p {
        font-size: 0.8rem;
        opacity: 0.95;
        font-weight: 300;
        margin: 0;
    }
    
    .input-section {
        background: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 10px;
    }
    
    .input-section h3 {
        color: #2c3e50;
        font-size: 0.95rem;
        margin: 0 0 8px 0;
        padding: 0;
        font-weight: 600;
    }
    
    .predict-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 8px solid #2e7d32;
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
        box-shadow: 0 5px 18px rgba(46, 125, 50, 0.15);
    }
    
    .predict-price {
        font-size: 2rem;
        font-weight: 800;
        color: #1b5e20;
        margin: 6px 0;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .sub-price {
        font-size: 0.8rem;
        color: #558b2f;
        margin-top: 3px;
        font-weight: 500;
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 8px;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
    }
    
    .info-card p {
        margin: 3px 0;
        font-size: 0.75rem;
        color: #2c3e50;
    }
    
    .info-card strong {
        color: #2e7d32;
        font-weight: 600;
    }
    
    .confidence-note {
        margin-top: 6px;
        padding: 6px;
        background: white;
        border-radius: 6px;
        color: #555;
        font-size: 0.7rem;
        border-left: 3px solid #ffc107;
    }
    
    .button-container {
        margin: 8px 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 8px 25px;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(46, 204, 113, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 18px rgba(46, 204, 113, 0.6);
    }
    
    .footer-text {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.7rem;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #ecf0f1;
    }
    
    .stSelectbox, .stSlider {
        margin-bottom: 4px;
    }
    
    </style>
""", unsafe_allow_html=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_model.pkl')
ENC_PATH   = os.path.join(BASE_DIR, 'models', 'encoders.pkl')
DATA_PATH  = os.path.join(BASE_DIR, 'data',   'wfp_food_prices_lka.csv')

@st.cache_resource
def load_assets():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENC_PATH, 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

@st.cache_data
def load_maps():
    df = pd.read_csv(DATA_PATH, skiprows=[1])
    df = df[['admin1', 'market', 'category', 'commodity', 'unit']].dropna()

    province_market = (
        df.groupby('admin1')['market']
        .unique().apply(sorted).to_dict()
    )
    category_commodity = (
        df.groupby('category')['commodity']
        .unique().apply(sorted).to_dict()
    )
    commodity_unit = (
        df.groupby('commodity')['unit']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    # New mappings for market-based filtering
    market_category = (
        df.groupby('market')['category']
        .unique().apply(sorted).to_dict()
    )
    market_category_commodity = {}
    for _, row in df.iterrows():
        market = row['market']
        category = row['category']
        commodity = row['commodity']
        key = (market, category)
        if key not in market_category_commodity:
            market_category_commodity[key] = set()
        market_category_commodity[key].add(commodity)
    # Convert sets to sorted lists
    market_category_commodity = {k: sorted(list(v)) for k, v in market_category_commodity.items()}
    
    return province_market, category_commodity, commodity_unit, market_category, market_category_commodity

try:
    model, encoders = load_assets()
    province_market_map, category_commodity_map, commodity_unit_map, market_category_map, market_category_commodity_map = load_maps()
except FileNotFoundError:
    st.error("⚠️ Model not found. Run both notebooks first!")
    st.stop()

def options(col):
    return sorted(list(encoders[col].classes_))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>🇱🇰 Sri Lanka Food Price Predictor</h1>
    <p>Predict 2026 food prices across Sri Lanka's markets</p>
</div>
""", unsafe_allow_html=True)

# ── Main Layout: Left Column (Inputs) | Right Column (Predictions) ────────────
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("""
    <div class="input-section">
        <h3>📋 Select Your Preferences</h3>
    </div>
    """, unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        province = st.selectbox("🗺️ Province", options('admin1'), key="province")
    
    with input_col2:
        available_markets = province_market_map.get(province, options('market'))
        market = st.selectbox("🏪 Market", available_markets, key="market")
    
    # Get categories available in the selected market
    with input_col1:
        available_categories = market_category_map.get(market, options('category'))
        category = st.selectbox("📦 Category", available_categories, key="category")
    
    # Get commodities for the selected market and category combination
    with input_col2:
        market_category_key = (market, category)
        available_commodities = market_category_commodity_map.get(market_category_key, options('commodity'))
        commodity = st.selectbox("🛒 Commodity", available_commodities, key="commodity")
    
    with input_col1:
        year = 2026  # Predicting for 2026 only
        month = st.selectbox(
            "📅 Month", list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2024, m, 1).strftime('%B'),
            key="month"
        )
    
    with input_col2:
        default_unit = commodity_unit_map.get(commodity, options('unit')[0])
        all_units    = options('unit')
        default_idx  = all_units.index(default_unit) if default_unit in all_units else 0
        unit = st.selectbox("⚖️ Unit", all_units, index=default_idx, key="unit")
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    predict_button = st.button("🔍 Predict Price", use_container_width=True, type="primary", key="predict_btn")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div class="input-section">
        <h3>💰 Price Prediction</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if predict_button:
        try:
            try:
                market_enc = encoders['market'].transform([market])[0]
            except ValueError:
                st.error(f"⚠️ Market '{market}' not found in training data.")
                st.stop()

            try:
                commodity_enc = encoders['commodity'].transform([commodity])[0]
            except ValueError:
                st.error(f"⚠️ Commodity '{commodity}' not found in training data.")
                st.stop()

            input_data = pd.DataFrame([{
                'year':          year,
                'month':         month,
                'admin1_enc':    encoders['admin1'].transform([province])[0],
                'market_enc':    market_enc,
                'category_enc':  encoders['category'].transform([category])[0],
                'commodity_enc': commodity_enc,
                'unit_enc':      encoders['unit'].transform([unit])[0],
            }])

            price      = model.predict(input_data)[0]
            month_name = pd.Timestamp(year, month, 1).strftime('%B %Y')

            # Price breakdown by unit
            unit_lower = unit.lower()
            if unit_lower == 'kg':
                extra = f"≈ LKR {price/10:,.2f} per 100g"
            elif unit_lower in ['l', 'litre', 'liter', 'lt']:
                extra = f"≈ LKR {price/10:,.2f} per 100ml"
            elif unit_lower in ['piece', 'pieces', 'each', 'unit', 'pcs']:
                extra = "↑ Price shown is per individual piece"
            else:
                extra = ""

            # Confidence for 2026 prediction
            confidence      = "🟡 Medium"
            confidence_note = "1 year ahead — reasonable estimate based on 2004-2025 data."

            st.markdown(f"""
            <div class="predict-box">
                <div style="color:#555;font-size:0.95rem;font-weight:500">Estimated price for</div>
                <div style="color:#1b5e20;font-size:1.1rem;font-weight:600;margin-bottom:10px">{month_name}</div>
                <div class="predict-price">₨ {price:,.0f}</div>
                <div class="sub-price">{extra}</div>
                <div class="confidence-note">
                    <strong>Confidence:</strong> {confidence}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#f0f8ff;padding:12px;border-radius:8px;margin-top:12px;border-left:3px solid #2196F3">
                <p style="font-size:0.85rem;color:#1565C0;margin:0"><strong>📊 Details:</strong></p>
                <p style="font-size:0.8rem;color:#424242;margin:5px 0 0 0">
                    {commodity} ({unit})<br>
                    {market}, {province}
                </p>
            </div>
            """, unsafe_allow_html=True)

            if year >= 2022:
                st.info("ℹ️ Post-2022 prices reflect economic crisis period.")

        except ValueError as e:
            st.error(f"⚠️ Could not predict. Try another combination. ({e})")
    else:
        st.markdown("""
        <div style="background:#f5f5f5;padding:20px;border-radius:8px;text-align:center;color:#666;margin-top:30px">
            <p style="font-size:0.95rem">👈 Select preferences on the left<br><strong>then click Predict</strong></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    <p>🌾 Data: World Food Programme — VAM Food Security Analysis, Sri Lanka</p>
</div>
""", unsafe_allow_html=True)