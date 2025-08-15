# app.py — Crashcaster (Streamlit Cloud–safe)
import os, json, time, pathlib, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# App config & light CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Crashcaster — Early Warning", page_icon="💥", layout="wide")
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
section[data-testid="stSidebar"] {background:#0E1117;}
div.block-container {padding-top: 1.0rem;}
.badge {display:inline-block;padding:6px 10px;border-radius:999px;font-weight:600;margin-right:6px;}
.badge.green{background:#1f6f43;color:#fff;}
.badge.orange{background:#a86b16;color:#fff;}
.badge.red{background:#a11f2c;color:#fff;}
.stDataFrame {overflow-x: auto;}
</style>
""", unsafe_allow_html=True)

def reason_badges(text: str):
    parts = [p.strip() for p in (text or "").split(";") if p.strip()]
    html = "".join([f"<span class='badge {'red' if ('High' in p or 'Large' in p) else 'green'}'>{p}</span>" for p in parts])
    return html or "<span class='badge green'>Stable</span>"

# -----------------------------------------------------------------------------
# Data config
# -----------------------------------------------------------------------------
BASE_URL_CG = "https://api.coingecko.com/api/v3"
CACHE_DIR = pathlib.Path("./cache"); CACHE_DIR.mkdir(exist_ok=True)
OFFLINE_DIR = pathlib.Path("./offline"); OFFLINE_DIR.mkdir(exist_ok=True)
OFFLINE_FILE = OFFLINE_DIR / "markets_sample.json"
TIMEOUT, REQUEST_DELAY, CACHE_TTL = 8, 2.0, 600  # small timeouts so cloud never hangs

def _cache_path(key): return CACHE_DIR / f"{key}.json"

def _read_cache(key, ttl=CACHE_TTL):
    p = _cache_path(key)
    if not p.exists(): return None
    if time.time() - p.stat().st_mtime > ttl: return None
    try: return json.loads(p.read_text())
    except: return None

def _write_cache(key, data): _cache_path(key).write_text(json.dumps(data))

def pct_change_from_sparkline(prices, hours):
    if not prices or len(prices) <= hours: return None
    last = float(prices[-1]); prev = float(prices[-hours-1])
    if prev == 0: return None
    return 100.0 * (last - prev) / prev

# -----------------------------------------------------------------------------
# Primary source (CoinGecko)
# -----------------------------------------------------------------------------
def fetch_cg_markets(max_coins=30, currency="usd", ttl=CACHE_TTL):
    key = f"cg_{currency}_{max_coins}_spark"
    cached = _read_cache(key, ttl=ttl)
    if cached is not None:
        return cached, "CACHE(CG)"
    url = (f"{BASE_URL_CG}/coins/markets?vs_currency={currency}"
           f"&order=market_cap_desc&per_page={max_coins}&page=1"
           f"&sparkline=true&price_change_percentage=1h,24h,7d,30d")
    r = requests.get(url, timeout=TIMEOUT, headers={"Accept": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"CG {r.status_code}")
    data = r.json()
    _write_cache(key, data)
    time.sleep(REQUEST_DELAY)  # polite
    return data, "LIVE(CG)"

def norm_from_cg(row):
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "symbol": (row.get("symbol") or "").upper(),
        "current_price": row.get("current_price"),
        "market_cap": row.get("market_cap"),
        "volume_24h": row.get("total_volume"),
        "price_change_percentage_1h": row.get("price_change_percentage_1h_in_currency", row.get("price_change_percentage_1h")),
        "price_change_percentage_24h": row.get("price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h")),
        "price_change_percentage_7d": row.get("price_change_percentage_7d_in_currency", row.get("price_change_percentage_7d")),
        "price_change_percentage_30d": row.get("price_change_percentage_30d_in_currency", row.get("price_change_percentage_30d")),
        "sparkline": (row.get("sparkline") or {}).get("price", []),
    }

def load_offline_json():
    if OFFLINE_FILE.exists():
        age = time.time() - OFFLINE_FILE.stat().st_mtime
        if age > 24 * 3600:
            st.warning("Offline data is over 24 hours old. Consider refreshing.")
        try: return json.loads(OFFLINE_FILE.read_text())
        except: return []
    return []

def get_markets_with_fallback(source="auto", max_coins=30, currency="usd"):
    """
    Returns: (df, src_used, last_error)
    """
    last_err = None
    # Try live if allowed
    if source in ("auto", "primary"):
        try:
            rows, src = fetch_cg_markets(max_coins, currency)
            norm = [norm_from_cg(r) for r in rows][:max_coins]
            # refresh offline snapshot (best-effort)
            try: OFFLINE_FILE.write_text(json.dumps(norm, indent=2))
            except: pass
            df = pd.DataFrame(norm)
            # compute 24/48/72 via sparkline
            df["price_change_percentage_48h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p, 48))
            df["price_change_percentage_72h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p, 72))
            df["price_change_percentage_24h"] = df.apply(
                lambda r: r["price_change_percentage_24h"] if pd.notna(r["price_change_percentage_24h"])
                else pct_change_from_sparkline(r["sparkline"], 24), axis=1)
            return df, src, None
        except Exception as e:
            last_err = e
    # Fallback to offline if present
    raw = load_offline_json()
    if raw:
        df = pd.DataFrame(raw).head(max_coins)
        if "sparkline" in df.columns:
            df["price_change_percentage_48h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p, 48))
            df["price_change_percentage_72h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p, 72))
            # compute 24h if missing
            if "price_change_percentage_24h" not in df.columns or df["price_change_percentage_24h"].isna().all():
                df["price_change_percentage_24h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p, 24))
        return df, "OFFLINE(JSON)", last_err
    # Nothing available
    return pd.DataFrame(), "NO_DATA", f"{last_err}"

# -----------------------------------------------------------------------------
# Features + risk
# -----------------------------------------------------------------------------
def _coerce_numeric(df):
    for c in ["current_price", "market_cap", "volume_24h",
              "price_change_percentage_1h", "price_change_percentage_24h",
              "price_change_percentage_48h", "price_change_percentage_72h",
              "price_change_percentage_7d", "price_change_percentage_30d"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _norm(s: pd.Series) -> pd.Series:
    s2 = s.replace([np.inf, -np.inf], np.nan)
    m, sd = s2.mean(), s2.std()
    if not np.isfinite(sd) or sd == 0: return pd.Series(0, index=s.index)
    z = (s2 - m) / sd
    return 1 / (1 + np.exp(-z))

def build_features_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = _coerce_numeric(df.copy())
    # Impute missing values
    df["price_change_percentage_1h"] = df["price_change_percentage_1h"].fillna(0)
    df["price_change_percentage_24h"] = df["price_change_percentage_24h"].fillna(0)
    df["abs_24h"] = df["price_change_percentage_24h"].abs()
    df["down_1h"] = df["price_change_percentage_1h"].clip(upper=0).abs()
    df["volume_ratio"] = (df["volume_24h"] / df["market_cap"]).clip(lower=0, upper=1)
    df["volatility_proxy"] = (
        df["price_change_percentage_1h"].abs().fillna(0) +
        df["price_change_percentage_24h"].abs().fillna(0) +
        df["price_change_percentage_7d"].abs().fillna(0)
    ) / 3.0
    risk = (0.50 * _norm(df["volatility_proxy"]) +
            0.30 * _norm(df["abs_24h"]) +
            0.15 * _norm(df["down_1h"]) +
            0.05 * _norm(df["volume_ratio"]))
    df["risk_score"] = (100 * risk).clip(0, 100).round(1)
    # Vectorized risk reasons
    med_vol = df["volatility_proxy"].median()
    med_abs24h = df["abs_24h"].median()
    df["risk_reason"] = np.where(df["volatility_proxy"] > med_vol, "High recent volatility; ", "")
    df["risk_reason"] += np.where(df["abs_24h"] > med_abs24h, "Large 24h move; ", "")
    df["risk_reason"] += np.where(df["price_change_percentage_1h"] < -0.5, "1h downside; ", "")
    df["risk_reason"] += np.where(df["volume_ratio"] > 0.5, "Heavy volume vs mcap; ", "")
    df["risk_reason"] = df["risk_reason"].str.rstrip("; ").replace("", "Stable / normal range")
    return df

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Data source", ["offline", "auto", "primary"], index=0)
    max_coins = st.slider("Coins", 10, 50, 30)  # Capped at 50 for performance
    threshold = st.slider("Risk threshold", 0, 100, 70)
    st.caption("Utilities")
    if st.button("Refresh offline snapshot"):
        try:
            rows, _ = fetch_cg_markets(max_coins=max_coins, currency="usd")
            norm = [norm_from_cg(r) for r in rows][:max_coins]
            OFFLINE_FILE.write_text(json.dumps(norm, indent=2))
            st.success("Offline snapshot refreshed ✓")
        except Exception as e:
            st.error(f"Live fetch failed: {e}")
    if st.button("Clear API cache"):
        for p in CACHE_DIR.glob("*.json"):
            try: p.unlink()
            except: pass
        st.success("Cache cleared.")

# -----------------------------------------------------------------------------
# Cached loader with safe fallbacks
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_and_score(src, n):
    df, src_used, last_err = get_markets_with_fallback(source=src, max_coins=n)
    if not df.empty:
        df = build_features_and_risk(df).sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df, src_used, last_err

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("💥 Crashcaster — Early Warning for Crypto Crashes")
st.markdown("Monitor cryptocurrency crash risks using real-time market data. Select a coin to analyze or explore high-risk assets below.")
with st.spinner("Loading and scoring coins..."):
    df, src_used, last_err = load_and_score(source, max_coins)

# If nothing to show, don’t crash the UI
if df.empty:
    st.error("No data available yet.")
    st.write("Try **Data source → offline** (after refreshing the offline snapshot) or **Data source → primary**.")
    if last_err: st.caption(f"Last error: {last_err}")
    st.stop()

# Format prices for readability
df["current_price"] = df["current_price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")

# KPIs
c1, c2, c3, c4 = st.columns(4)
top = df.iloc[0]
c1.metric("Top Risk Coin", f"{top['symbol']}", f"{top['risk_score']:.0f}")
c2.metric("Market Risk Avg", f"{df['risk_score'].mean():.0f}%")
c3.metric("Coins ≥ Threshold", f"{(df['risk_score'] >= threshold).sum()}/{len(df)}")
c4.metric("Data Source", src_used)
if src_used.startswith("OFFLINE") and last_err:
    st.warning(f"Using offline snapshot. Live fetch previously failed with: {last_err}")

# Hero “Try it now”
st.markdown("### Try it now")
left, right = st.columns([3, 1])
with left:
    symbols = sorted([s for s in df["symbol"].dropna().unique().tolist() if s]) or ["N/A"]
    selected_symbol = st.selectbox("Pick a coin", symbols, index=0 if symbols[0] != "N/A" else None,
                                  disabled=len(symbols) == 1 and symbols[0] == "N/A")
with right:
    analyze_clicked = st.button("Analyze", use_container_width=True)
if analyze_clicked and symbols and symbols[0] != "N/A":
    with st.spinner("Analyzing..."):
        st.session_state["selected_symbol"] = selected_symbol
if "selected_symbol" in st.session_state:
    sel = st.session_state["selected_symbol"]
    row = df[df["symbol"] == sel].head(1)
    if not row.empty:
        r = row.iloc[0]
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            st.subheader(f"{r['name']} ({r['symbol']})")
            spark = r.get("sparkline", None)
            if isinstance(spark, (list, tuple)) and len(spark) > 5:
                fig = go.Figure(go.Scatter(y=spark, mode="lines", line=dict(color="#1f77b4")))
                fig.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis_title="Last 7 Days",
                    yaxis_title="Price (USD)"
                )
                st.plotly_chart(fig, use_container_width=True)
            s24 = r.get("price_change_percentage_24h", 0) or 0
            s48 = r.get("price_change_percentage_48h", np.nan)
            s72 = r.get("price_change_percentage_72h", np.nan)
            s48_txt = f"{s48:+.2f}%" if pd.notna(s48) else "—"
            s72_txt = f"{s72:+.2f}%" if pd.notna(s72) else "—"
            st.markdown(f"**24h:** {s24:+.2f}% · **48h:** {s48_txt} · **72h:** {s72_txt}")
        with cc2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(r["risk_score"]),
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff4d4d" if r["risk_score"] >= threshold else "#1f6f43"},
                    "steps": [
                        {"range": [0, 50], "color": "#e6f3e6"},
                        {"range": [50, 75], "color": "#fff4e6"},
                        {"range": [75, 100], "color": "#ffe6e6"}
                    ]
                },
                title={"text": "Risk Score"}
            ))
            gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(f"<div>{reason_badges(r['risk_reason'])}</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "✅ Recommendations"])
with tab1:
    st.subheader("At-Risk Coins")
    danger = df[df["risk_score"] >= threshold].copy()
    danger["risk_score"] = danger["risk_score"].round(0).astype("Int64")
    st.dataframe(danger[["name", "symbol", "current_price", "price_change_percentage_24h", "risk_score", "risk_reason"]],
                 use_container_width=True)
    st.subheader("All Coins")
    all_df = df.copy()
    all_df["risk_score"] = all_df["risk_score"].round(0).astype("Int64")
    st.dataframe(all_df[["name", "symbol", "current_price", "price_change_percentage_24h", "risk_score", "risk_reason"]],
                 use_container_width=True)
    st.subheader("Charts")
    # A) Top-10 by risk
    top_risk = df.nlargest(10, "risk_score").sort_values("risk_score", ascending=True)
    fig1 = px.bar(top_risk, x="risk_score", y="symbol", orientation="h",
                  title="Top-10 Crash Risk (higher = riskier)",
                  labels={"risk_score": "Risk Score (0–100)", "symbol": "Coin"})
    fig1.add_vline(x=threshold, line_dash="dash", line_width=2)
    fig1.update_layout(font=dict(size=12), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig1, use_container_width=True)
    # B) Risk distribution + threshold line
    fig_hist = px.histogram(df, x="risk_score", nbins=20,
                            title="Risk Score Distribution",
                            labels={"risk_score": "Risk Score (0–100)", "count": "Number of Coins"})
    fig_hist.add_vline(x=threshold, line_dash="dash", line_width=2)
    fig_hist.update_layout(font=dict(size=12), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"Vertical dashed line shows the risk threshold ({threshold}) set in the sidebar.")
    # C) Top-10 24h movers
    movers = df.assign(abs24=df["price_change_percentage_24h"].abs()) \
               .nlargest(10, "abs24") \
               .sort_values("price_change_percentage_24h", ascending=True)
    fig_moves = px.bar(movers, x="price_change_percentage_24h", y="symbol", orientation="h",
                       title="Top-10 24h Movers (±)",
                       labels={"price_change_percentage_24h": "24h %", "symbol": "Coin"})
    fig_moves.update_layout(font=dict(size=12), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_moves, use_container_width=True)

with tab2:
    def recommend_coins_local(dff, strategy="trend", top_k=10):
        d = dff.copy()
        for col in ["price_change_percentage_24h", "price_change_percentage_48h",
                    "price_change_percentage_72h", "volatility_proxy", "risk_score"]:
            if col not in d.columns: d[col] = np.nan
            d[col] = pd.to_numeric(d[col], errors="coerce")
        c24 = d["price_change_percentage_24h"].fillna(0)
        c48 = d["price_change_percentage_48h"].fillna(0)
        c72 = d["price_change_percentage_72h"].fillna(0)
        vol = d["volatility_proxy"].fillna(d["volatility_proxy"].median())
        def ok_nonneg_or_nan(s): return (s >= 0) | (s.isna())
        if strategy == "trend":
            mask = ((d["risk_score"] < 65) & (vol < vol.quantile(0.7)) &
                    (c24 > 0) & ok_nonneg_or_nan(d["price_change_percentage_48h"]) &
                    ok_nonneg_or_nan(d["price_change_percentage_72h"]))
            score = (0.60 * c24 + 0.25 * c48 + 0.15 * c72 - 0.20 * vol)
        else:  # reversal
            mask = ((d["risk_score"] < 55) & (c24 < 0) &
                    ok_nonneg_or_nan(d["price_change_percentage_72h"]) &
                    (vol < vol.quantile(0.8)))
            score = ((-1.0) * c24 + 0.30 * c72 - 0.20 * vol)
        out = d[mask].copy()
        out["recommend_score"] = score[mask]
        out = out.sort_values("recommend_score", ascending=False).head(top_k)
        if out.empty:
            fb = d[(d["risk_score"] < 60)].copy()
            fb["recommend_score"] = 0.70 * c24 - 0.20 * vol
            out = fb.sort_values("recommend_score", ascending=False).head(top_k)
            st.info("Showing fallback list (limited 48/72h data). Refresh the offline snapshot when online to improve results.")
        cols = ["name", "symbol", "current_price", "price_change_percentage_24h",
                "price_change_percentage_48h", "price_change_percentage_72h",
                "volatility_proxy", "risk_score", "risk_reason", "recommend_score"]
        return out[cols]

    strat = st.selectbox("Strategy", ["trend", "reversal"], index=0)
    rec = recommend_coins_local(df, strat, top_k=10)
    rec["current_price"] = rec["current_price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    st.dataframe(rec, use_container_width=True)

st.caption("Tip: If the API hiccups, use the sidebar to create an offline snapshot and switch Data source → 'offline'. Not financial advice.")
