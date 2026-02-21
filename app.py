import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Sri Lanka Food Price Predictor",
                   page_icon="📊", layout="wide")

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
        background: #f8fafb;
        padding: 15px !important;
    }
    
    .header-container {
        background: #0f766e;
        color: white;
        padding: 16px 14px;
        border-radius: 10px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.15);
    }
    
    .header-container h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    
    .header-container p {
        font-size: 0.75rem;
        opacity: 0.95;
        font-weight: 300;
        margin: 0;
    }
    
    .input-section {
        background: white;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(15, 118, 110, 0.08);
        margin-bottom: 14px;
        border: 1px solid #d1fae5;
    }
    
    .input-section h3 {
        color: #0f766e;
        font-size: 0.95rem;
        margin: 0 0 10px 0;
        padding: 0;
        font-weight: 600;
    }
    
    .predict-box {
        background: #f0fdfa;
        border-left: 5px solid #0f766e;
        border-radius: 8px;
        padding: 14px;
        margin-top: 12px;
        box-shadow: 0 1px 3px rgba(15, 118, 110, 0.1);
    }
    
    .predict-price {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0d7a7a;
        margin: 8px 0;
        text-shadow: none;
    }
    
    .sub-price {
        font-size: 0.8rem;
        color: #0f766e;
        margin-top: 4px;
        font-weight: 500;
    }
    
    .info-card {
        background: #f0fdfa;
        padding: 6px;
        border-radius: 8px;
        border-left: 4px solid #0f766e;
    }
    
    .info-card p {
        margin: 2px 0;
        font-size: 0.7rem;
        color: #134e4a;
    }
    
    .info-card strong {
        color: #0f766e;
        font-weight: 600;
    }
    
    .confidence-note {
        margin-top: 8px;
        padding: 8px;
        background: white;
        border-radius: 6px;
        color: #134e4a;
        font-size: 0.7rem;
        border-left: 3px solid #0f766e;
    }
    
    .button-container {
        margin: 12px 0;
    }
    
    .stButton > button {
        background: #0f766e;
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 6px 20px;
        border: none;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(15, 118, 110, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(15, 118, 110, 0.3);
        background: #0d7a7a;
    }
    
    .footer-text {
        text-align: center;
        color: #0f766e;
        font-size: 0.7rem;
        margin-top: 20px;
        padding-top: 12px;
        border-top: 1px solid #d1fae5;
    }
    
    .stSelectbox, .stSlider {
        margin-bottom: 8px;
    }
    
    [data-testid="stVerticalBlock"] {
        gap: 1.2rem;
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
    df = df[['admin1', 'market', 'category', 'commodity', 'unit', 'latitude', 'longitude']].dropna()

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
    
    # Market coordinates mapping
    market_coords = df.groupby('market')[['latitude', 'longitude']].first().to_dict()
    
    return province_market, category_commodity, commodity_unit, market_category, market_category_commodity, market_coords

try:
    model, encoders = load_assets()
    province_market_map, category_commodity_map, commodity_unit_map, market_category_map, market_category_commodity_map, market_coords_map = load_maps()
except FileNotFoundError:
    st.error("⚠️ Model not found. Run both notebooks first!")
    st.stop()

def options(col):
    return sorted(list(encoders[col].classes_))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>Sri Lanka Food Price Predictor</h1>
    <p>Predict 2026 food prices across Sri Lanka's markets</p>
</div>
""", unsafe_allow_html=True)

# ── Main Layout: Left Column (Inputs) | Right Column (Predictions) ────────────
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("""
    <div class="input-section">
        <h3>Preferences</h3>
    </div>
    """, unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        province = st.selectbox("Province", options('admin1'), key="province")
    
    with input_col2:
        available_markets = province_market_map.get(province, options('market'))
        market = st.selectbox("Market", available_markets, key="market")
    
    # Get categories available in the selected market
    with input_col1:
        available_categories = market_category_map.get(market, options('category'))
        category = st.selectbox("Category", available_categories, key="category")
    
    # Get commodities for the selected market and category combination
    with input_col2:
        market_category_key = (market, category)
        available_commodities = market_category_commodity_map.get(market_category_key, options('commodity'))
        commodity = st.selectbox("Commodity", available_commodities, key="commodity")
    
    with input_col1:
        year = 2026  # Predicting for 2026 only
        month = st.selectbox(
            "Month (2026)", list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2024, m, 1).strftime('%B'),
            key="month"
        )
    
    with input_col2:
        default_unit = commodity_unit_map.get(commodity, options('unit')[0])
        all_units    = options('unit')
        default_idx  = all_units.index(default_unit) if default_unit in all_units else 0
        unit = st.selectbox("Unit", all_units, index=default_idx, key="unit")
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    predict_button = st.button("Predict Price", use_container_width=True, type="primary", key="predict_btn")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div class="input-section">
        <h3>Price Prediction</h3>
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

            # Get market coordinates for geographic features
            if market in market_coords_map.get('latitude', {}):
                market_lat = market_coords_map['latitude'][market]
                market_lon = market_coords_map['longitude'][market]
            else:
                market_lat, market_lon = 0.0, 0.0  # Default if not found


            input_data = pd.DataFrame([{
                'year':          year,
                'month':         month,
                'latitude':      market_lat,
                'longitude':     market_lon,
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
                <div style="color:#134e4a;font-size:0.95rem;font-weight:500;margin-bottom:8px">Estimated price for</div>
                <div style="color:#0f766e;font-size:1.1rem;font-weight:600;margin-bottom:12px">{month_name}</div>
                <div class="predict-price">₨ {price:,.0f}</div>
                <div class="sub-price">{extra}</div>
                <div class="confidence-note">
                    <strong>Confidence:</strong> {confidence}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#f0fdfa;padding:12px;border-radius:8px;margin-top:12px;border-left:3px solid #0f766e">
                <p style="font-size:0.85rem;color:#0f766e;margin:0;font-weight:600">Details</p>
                <p style="font-size:0.8rem;color:#134e4a;margin:8px 0 0 0">
                    {commodity} ({unit})<br>
                    {market}, {province}
                </p>
            </div>
            """, unsafe_allow_html=True)

            if year >= 2022:
                st.info("Post-2022 prices reflect crisis period.")

        except ValueError as e:
            st.error(f"Could not predict. Try another combination. ({e})")
    else:
        st.markdown("""
        <div style="background:#f0fdfa;padding:25px;border-radius:8px;text-align:center;color:#0f766e;margin-top:25px;border:1px solid #d1fae5">
            <p style="font-size:1rem;font-weight:500">Select options and click Predict</p>
        </div>
        """, unsafe_allow_html=True)

# ── Model Insights (Below Columns) ─────────────────────────────────────────────
if predict_button:
    try:
        st.markdown("""
        <div style="margin-top:20px;padding:14px;background:#f0fdfa;border-radius:8px;border-left:4px solid #0f766e">
            <p style="margin:0;font-size:0.9rem;color:#0f766e;font-weight:600">Model Insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View Feature Importance & Model Accuracy"):
            col_shap1, col_shap2 = st.columns(2)
            
            with col_shap1:
                shap_path = os.path.join(BASE_DIR, 'outputs', 'shap_summary.png')
                if os.path.exists(shap_path):
                    try:
                        st.image(shap_path, caption="Feature Importance (SHAP)")
                    except Exception as e:
                        st.caption("SHAP visualization")
            
            with col_shap2:
                acc_path = os.path.join(BASE_DIR, 'outputs', 'actual_vs_predicted.png')
                if os.path.exists(acc_path):
                    try:
                        st.image(acc_path, caption="Model Accuracy (Test Set)")
                    except Exception as e:
                        st.caption("Accuracy plot")
    except:
        pass

st.markdown("""
<div class="footer-text">
    <p>Data: World Food Programme — VAM Food Security Analysis, Sri Lanka</p>
</div>
""", unsafe_allow_html=True)