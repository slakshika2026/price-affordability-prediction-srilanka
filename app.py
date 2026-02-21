import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="🇱🇰 Sri Lanka Food Price Predictor",
                   page_icon="🥗", layout="centered")

st.markdown("""
    <style>
    .predict-box {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .predict-price { font-size: 2.4rem; font-weight: 700; color: #1b5e20; }
    .sub-price { font-size: 1rem; color: #555; margin-top: 4px; }
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
    return province_market, category_commodity, commodity_unit

try:
    model, encoders = load_assets()
    province_market_map, category_commodity_map, commodity_unit_map = load_maps()
except FileNotFoundError:
    st.error("⚠️ Model not found. Run both notebooks first!")
    st.stop()

def options(col):
    return sorted(list(encoders[col].classes_))

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🇱🇰 Sri Lanka Food Price Predictor")
st.markdown("Estimate the **retail price** of essential food commodities across Sri Lanka.")
st.markdown("---")
st.subheader("📋 Select Location & Commodity")

col1, col2 = st.columns(2)

with col1:
    province = st.selectbox("🗺️ Province", options('admin1'))

    available_markets = province_market_map.get(province, options('market'))
    market = st.selectbox("🏪 Market", available_markets)

    category = st.selectbox("📦 Category", options('category'))

with col2:
    available_commodities = category_commodity_map.get(category, options('commodity'))
    commodity = st.selectbox("🛒 Commodity", available_commodities)

    default_unit = commodity_unit_map.get(commodity, options('unit')[0])
    all_units    = options('unit')
    default_idx  = all_units.index(default_unit) if default_unit in all_units else 0
    unit = st.selectbox("⚖️ Unit", all_units, index=default_idx)

    year = 2026  # Predicting for 2026 only
    month = st.selectbox(
        "📅 Month", list(range(1, 13)),
        format_func=lambda m: pd.Timestamp(2024, m, 1).strftime('%B')
    )

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("")
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
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
            <div style="color:#555;font-size:0.9rem">Estimated retail price for {month_name}</div>
            <div class="predict-price">LKR {price:,.2f}
                <span style="font-size:1rem;color:#555"> per {unit}</span>
            </div>
            <div class="sub-price">{extra}</div>
            <div style="margin-top:8px;font-size:0.85rem;color:#666">
                Confidence: {confidence} — {confidence_note}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"- **Commodity:** {commodity}")
            st.markdown(f"- **Category:** {category}")
            st.markdown(f"- **Unit:** {unit}")
        with c2:
            st.markdown(f"- **Province:** {province}")
            st.markdown(f"- **Market:** {market}")
            st.markdown(f"- **Period:** {month_name}")

        if year >= 2022:
            st.info("ℹ️ Post-2022 prices reflect Sri Lanka's economic crisis period.")

    except ValueError as e:
        st.error(f"⚠️ Could not predict. Try another combination. ({e})")

st.markdown("---")
st.caption("Predicting 2026 prices only. Data: World Food Programme — VAM Food Security Analysis, Sri Lanka (2004–2025)")