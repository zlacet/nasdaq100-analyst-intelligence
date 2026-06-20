import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import warnings
import requests
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyst Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #0f2d5e; font-weight: 800; }
    h2, h3 { color: #0f2d5e; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1a56a0; }
    .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
    .predict-result {
        background: #0f2d5e;
        color: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .predict-result .pct { font-size: 3rem; font-weight: 800; }
    .predict-result .sublabel { font-size: 0.95rem; opacity: 0.8; margin-top: 0.3rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .macro-pill {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56a0;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RATING_MAP = {
    "buy": "Buy", "outperform": "Buy", "overweight": "Buy",
    "positive": "Buy", "strong buy": "Buy", "strongbuy": "Buy",
    "top pick": "Buy", "toppick": "Buy", "conviction buy": "Buy",
    "sector outperform": "Buy", "outperformer": "Buy",
    "hold": "Hold", "neutral": "Hold", "equal weight": "Hold",
    "market perform": "Hold", "sector perform": "Hold",
    "in-line": "Hold", "inline": "Hold", "peer perform": "Hold",
    "sell": "Sell", "underperform": "Sell", "underweight": "Sell",
    "reduce": "Sell", "negative": "Sell",
}
RATING_NUM = {"Buy": 3, "Hold": 2, "Sell": 1}

# ── Fetch live macro data ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_macro():
    """Fetch current VIX and Fed Funds Rate. Falls back to recent averages."""
    try:
        import yfinance as yf
        vix_data = yf.download("^VIX", period="5d", progress=False)
        vix_data.columns = vix_data.columns.get_level_values(0)
        current_vix = float(vix_data["Close"].iloc[-1])
    except Exception:
        current_vix = 18.5  # fallback: recent average

    try:
        # FRED API — free, no key needed for this endpoint
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        fed_df = pd.read_csv(url, parse_dates=["DATE"])
        current_fed = float(fed_df["FEDFUNDS"].iloc[-1])
    except Exception:
        current_fed = 5.25  # fallback: recent rate

    return current_vix, current_fed

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("anachart.csv")
    df["rating_clean"] = df["rating_post"].apply(
        lambda r: RATING_MAP.get(r.lower().strip()) if pd.notna(r) else None
    )
    df["rating_prior_clean"] = df["rating_prior"].apply(
        lambda r: RATING_MAP.get(r.lower().strip()) if pd.notna(r) else None
    )
    df["year"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False).dt.year
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df["price_target_post"] = pd.to_numeric(df["price_target_post"], errors="coerce")
    return df[df["year"].between(2014, 2025)].copy()

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    encoder = joblib.load("ticker_encoder.pkl")
    return model, encoder

df = load_data()
model, ticker_encoder = load_model()
current_vix, current_fed = fetch_macro()
all_tickers = sorted(ticker_encoder.classes_.tolist())
all_brokers = sorted(df["broker_number"].dropna().astype(int).unique().tolist())

# ── Pre-compute datasets ──────────────────────────────────────────────────────
@st.cache_data
def get_yearly(ticker_filter=None):
    d = df if ticker_filter is None else df[df["ticker"] == ticker_filter]
    yearly = d.groupby(["year", "rating_clean"]).size().unstack(fill_value=0)
    for col in ["Buy", "Hold", "Sell"]:
        if col not in yearly.columns:
            yearly[col] = 0
    yearly["total"]    = yearly[["Buy", "Hold", "Sell"]].sum(axis=1)
    yearly["buy_pct"]  = yearly["Buy"]  / yearly["total"] * 100
    yearly["hold_pct"] = yearly["Hold"] / yearly["total"] * 100
    yearly["sell_pct"] = yearly["Sell"] / yearly["total"] * 100
    yearly = yearly.reset_index()
    yearly["year"] = yearly["year"].astype(int)
    return yearly

@st.cache_data
def get_broker_stats(ticker_filter=None):
    d = df if ticker_filter is None else df[df["ticker"] == ticker_filter]
    broker_stats = (
        d.dropna(subset=["broker_number"])
        .groupby(["broker_number", "rating_clean"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["Buy", "Hold", "Sell"]:
        if col not in broker_stats.columns:
            broker_stats[col] = 0
    broker_stats["total"]   = broker_stats[["Buy", "Hold", "Sell"]].sum(axis=1)
    broker_stats            = broker_stats[broker_stats["total"] >= 50]
    broker_stats["buy_pct"] = broker_stats["Buy"] / broker_stats["total"] * 100
    broker_stats            = broker_stats.sort_values("buy_pct", ascending=False).reset_index()
    broker_stats["broker_number"] = broker_stats["broker_number"].astype(int).astype(str)
    return broker_stats

@st.cache_data
def get_pt_by_year(ticker_filter=None):
    d = df if ticker_filter is None else df[df["ticker"] == ticker_filter]
    pt = d.dropna(subset=["close_price", "price_target_post"]).copy()
    pt = pt[pt["close_price"] > 0]
    pt["implied_upside"] = (pt["price_target_post"] - pt["close_price"]) / pt["close_price"] * 100
    pt = pt[pt["implied_upside"].between(-100, 200)]
    pt_by_year = pt.groupby("year")["implied_upside"].agg(["mean", "median"]).reset_index()
    pt_by_year["year"] = pt_by_year["year"].astype(int)
    return pt_by_year

@st.cache_data
def get_updown(ticker_filter=None):
    d = df.copy() if ticker_filter is None else df[df["ticker"] == ticker_filter].copy()
    d["rating_num"]       = d["rating_clean"].map(RATING_NUM)
    d["rating_prior_num"] = d["rating_prior_clean"].map(RATING_NUM)
    changes = d.dropna(subset=["rating_num", "rating_prior_num", "year"]).copy()
    changes["direction"] = changes.apply(
        lambda r: "Upgrade"   if r["rating_num"] > r["rating_prior_num"]
        else ("Downgrade" if r["rating_num"] < r["rating_prior_num"] else "Hold"),
        axis=1
    )
    updown = changes.groupby(["year", "direction"]).size().unstack(fill_value=0).reset_index()
    for col in ["Upgrade", "Downgrade", "Hold"]:
        if col not in updown.columns:
            updown[col] = 0
    updown["year"] = updown["year"].astype(int)
    return updown

def get_broker_ranking(ticker, year, vix, fed):
    ticker_encoded = ticker_encoder.transform([ticker])[0]
    results = []
    for b in all_brokers:
        pred = model.predict([[b, ticker_encoded, vix, fed, year]])[0] * 100
        results.append({"Broker": str(b), "pred": pred})
    results = sorted(results, key=lambda x: x["pred"], reverse=True)
    return pd.DataFrame(results)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Analyst Intelligence Dashboard")
st.markdown("**AnaChart NASDAQ-100 Dataset · 2014–2025 · ALY6980 Capstone | Lacet & Kuppili**")

st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
total_ratings   = int(df["rating_clean"].notna().sum())
unique_brokers  = int(df["broker_number"].nunique())
unique_tickers  = int(df["ticker"].nunique())
overall_buy_pct = df[df["rating_clean"].notna()]["rating_clean"].eq("Buy").mean() * 100

c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, f"{total_ratings:,}", "Total Ratings Analyzed"),
    (c2, f"{unique_brokers}",  "Unique Brokers"),
    (c3, f"{unique_tickers}",  "NASDAQ-100 Tickers"),
    (c4, f"{overall_buy_pct:.1f}%", "Overall Buy Rate"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="value">{val}</div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Rating Trends",
    "🏦 Broker Behavior",
    "🎯 Price Target Aggressiveness",
    "🤖 Predicted Implied Upside / Downside",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 (first) — Broker Ranking & Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    import datetime
    current_year = datetime.datetime.now().year

    st.markdown("### Predicted Implied Upside / Downside")
    st.info("Enter a broker and ticker, then adjust the market conditions. **VIX** measures market volatility and fear — a higher VIX signals uncertainty, which tends to make brokers more conservative with price targets. **Fed Rate** reflects the cost of borrowing — higher rates compress valuations, pushing implied upside down. Together these macro conditions shape how aggressively a broker will set their price target.")
    st.caption(f"Model: XGBoost · R² = 0.888 · Predictions based on {datetime.datetime.now().strftime('%A, %B %d, %Y')}")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("#### Inputs")
        rank_ticker = st.selectbox(
            "Ticker",
            options=all_tickers,
            index=all_tickers.index("AAPL") if "AAPL" in all_tickers else 0,
            key="rank_ticker",
        )
        single_broker = st.selectbox(
            "Broker ID",
            options=all_brokers,
            index=all_brokers.index(29901) if 29901 in all_brokers else 0,
            key="single_broker",
        )
        user_vix = st.slider(
            "VIX (Market Volatility)",
            min_value=5.0,
            max_value=80.0,
            value=round(current_vix, 1),
            step=0.1,
            help="Current VIX is ~16.78. Higher VIX = more market fear.",
            key="user_vix",
        )
        user_fed = st.slider(
            "Fed Funds Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.63,
            step=0.01,
            help="Current Fed Rate is ~3.63%.",
            key="user_fed",
        )
        st.markdown(f"📅 **{datetime.datetime.now().strftime('%A, %B %d, %Y')}**")
        lookup_btn = st.button("🔮 Predict", use_container_width=True, type="primary")

    with col_out:
        st.markdown("#### Output")
        if lookup_btn:
            try:
                te   = ticker_encoder.transform([rank_ticker])[0]
                pred = model.predict([[int(single_broker), te, user_vix, user_fed, current_year]])[0] * 100
                color = "#16a34a" if pred >= 0 else "#dc2626"
                sign  = "+" if pred >= 0 else ""
                st.markdown(f"""
                <div class="predict-result">
                    <div style="font-size:0.9rem; opacity:0.8; margin-bottom:0.4rem;">Predicted Implied Upside</div>
                    <div class="pct" style="color:{color}">{sign}{pred:.1f}%</div>
                    <div class="sublabel">Broker {single_broker} · {rank_ticker} · VIX {user_vix:.1f} · Fed {user_fed:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                direction = "above" if pred >= 0 else "below"
                st.markdown(
                    f"\nBroker **{single_broker}** is predicted to set a **{rank_ticker}** "
                    f"price target **{sign}{pred:.1f}%** {direction} the current stock price "
                    f"(VIX: {user_vix:.1f}, Fed Rate: {user_fed:.2f}%)."
                )
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div style="background:#f1f5f9; border-radius:12px; padding:2rem; text-align:center; color:#94a3b8; height:180px; display:flex; align-items:center; justify-content:center;">
                <div>
                    <div style="font-size:2.5rem">🔮</div>
                    <div style="margin-top:0.5rem">Select inputs and click Predict</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Rating Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    ticker_filter_t1 = st.selectbox("Filter by Ticker", ["All Tickers"] + all_tickers, key="t1")
    tf1 = None if ticker_filter_t1 == "All Tickers" else ticker_filter_t1
    yearly = get_yearly(tf1)

    st.markdown("### Buy Rating % Over Time")
    st.caption("Core trend: Buy% declined from 83.5% in 2014 to 72.4% in 2025 — a 12-point drop over 11 years")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["buy_pct"],
        mode="lines+markers",
        line=dict(color="#1a56a0", width=3),
        marker=dict(size=7),
        name="Buy %",
    ))
    if len(yearly) > 1:
        z = np.polyfit(yearly["year"], yearly["buy_pct"], 1)
        p_fn = np.poly1d(z)
        fig_line.add_trace(go.Scatter(
            x=yearly["year"], y=p_fn(yearly["year"]),
            mode="lines", line=dict(color="#90b8e0", width=2, dash="dash"),
            name="Trend",
        ))
    fig_line.update_layout(
        xaxis_title="Year", yaxis_title="Buy %",
        yaxis=dict(ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    updown = get_updown(tf1)
    st.markdown("### Upgrades, Holds & Downgrades by Year")
    st.caption("Upgrades outpaced downgrades in 9 of 11 years — confirming persistent bullish bias")

    fig_updown = go.Figure()
    fig_updown.add_trace(go.Bar(
        x=updown["year"], y=updown["Upgrade"],
        name="Upgrades", marker_color="#16a34a", opacity=0.85, offsetgroup=0,
    ))
    fig_updown.add_trace(go.Bar(
        x=updown["year"], y=updown["Downgrade"],
        name="Downgrades", marker_color="#dc2626", opacity=0.85, offsetgroup=1,
    ))
    fig_updown.update_layout(
        barmode="group",
        xaxis_title="Year", yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400,
    )
    st.plotly_chart(fig_updown, use_container_width=True)
    st.info("Analyst behavior has shifted dramatically over 11 years. The model learned these time-based patterns — making year a critical driver of how aggressively brokers set price targets.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Broker Behavior
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    ticker_filter_t2 = st.selectbox("Filter by Ticker", ["All Tickers"] + all_tickers, key="t2")
    tf2 = None if ticker_filter_t2 == "All Tickers" else ticker_filter_t2
    broker_stats = get_broker_stats(tf2)
    avg_buy      = broker_stats["buy_pct"].mean()
    top10        = broker_stats.head(10).sort_values("buy_pct")
    bottom10     = broker_stats.tail(10).sort_values("buy_pct")

    st.caption(f"Min 50 ratings per broker · Avg Buy%: {avg_buy:.0f}%")

    st.markdown("### Top 10 Most Bullish Brokers")
    fig_top = go.Figure(go.Bar(
        x=top10["buy_pct"], y=top10["broker_number"],
        orientation="h", marker_color="#16a34a", opacity=0.88, width=0.6,
    ))
    fig_top.add_vline(x=avg_buy, line_dash="dash", line_color="#6b7280",
                      annotation_text=f"Avg {avg_buy:.0f}%", annotation_position="top right")
    fig_top.update_layout(
        xaxis_title="Buy %", yaxis_title="Broker ID",
        xaxis=dict(ticksuffix="%", range=[0, 105]),
        yaxis=dict(type="category", tickfont=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, margin=dict(l=100),
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("### Bottom 10 Least Bullish Brokers")
    fig_bot = go.Figure(go.Bar(
        x=bottom10["buy_pct"], y=bottom10["broker_number"],
        orientation="h", marker_color="#dc2626", opacity=0.88, width=0.6,
    ))
    fig_bot.add_vline(x=avg_buy, line_dash="dash", line_color="#6b7280",
                      annotation_text=f"Avg {avg_buy:.0f}%", annotation_position="top right")
    fig_bot.update_layout(
        xaxis_title="Buy %", yaxis_title="Broker ID",
        xaxis=dict(ticksuffix="%", range=[0, 105]),
        yaxis=dict(type="category", tickfont=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, margin=dict(l=100),
    )
    st.plotly_chart(fig_bot, use_container_width=True)

    st.info("A Buy rating from broker 26793 means something very different than one from broker 24503. The model accounts for each broker's historical bias when predicting implied upside.")

    with st.expander("📋 Full broker table"):
        display = broker_stats[["broker_number", "Buy", "Hold", "Sell", "total", "buy_pct"]].copy()
        display.columns = ["Broker ID", "Buy", "Hold", "Sell", "Total", "Buy %"]
        display["Buy %"] = display["Buy %"].map("{:.1f}%".format)
        st.dataframe(display, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Price Target Aggressiveness
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    ticker_filter_t3 = st.selectbox("Filter by Ticker", ["All Tickers"] + all_tickers, key="t3")
    tf3 = None if ticker_filter_t3 == "All Tickers" else ticker_filter_t3
    pt_by_year = get_pt_by_year(tf3)

    st.markdown("### Price Target Implied Upside by Year")
    st.caption("Target vs close price at time of rating · Blue bars = mean · Orange line = median")

    fig_pt = go.Figure()
    fig_pt.add_trace(go.Bar(
        x=pt_by_year["year"], y=pt_by_year["mean"],
        name="Mean", marker_color="#0369a1", opacity=0.85,
    ))
    fig_pt.add_trace(go.Scatter(
        x=pt_by_year["year"], y=pt_by_year["median"],
        mode="lines+markers", name="Median",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=7),
    ))
    fig_pt.add_hline(y=0, line_color="#6b7280", line_dash="dot")
    fig_pt.update_layout(
        xaxis_title="Year", yaxis_title="Implied Upside %",
        yaxis=dict(ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400,
    )
    st.plotly_chart(fig_pt, use_container_width=True)

    st.info("Implied upside is exactly what our model predicts. This chart shows the historical distribution the model was trained on — how far above or below the stock price analysts have historically set their targets.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.info("**2014–2021:** Mean was negative — large outliers dragged it down. Median crossed 0% around 2016–17, meaning the typical analyst was already bullish.")
    with col_r:
        st.info("**Post-2022:** Mean and median both positive and converging — analysts became more consistently bullish in target-setting across the board.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("ALY6980 Capstone · Lacet & Kuppili · AnaChart NASDAQ-100 Dataset 2014–2025 · XGBoost Model R²=0.888")
