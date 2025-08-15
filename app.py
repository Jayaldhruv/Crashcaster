import os, json, time, pathlib, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------- App config --------
st.set_page_config(page_title="Crashcaster — Early Warning", page_icon="💥", layout="wide")

# --- UX polish (hide chrome, badges) ---
HIDE_DEFAULT = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
section[data-testid="stSidebar"] {background:#0E1117;}
div.block-container {padding-top: 1.0rem;}
.badge {display:inline-block;padding:6px 10px;border-radius:999px;font-weight:600;margin-right:6px;}
.badge.green{background:#1f6f43;color:#fff;}
.badge.orange{background:#a86b16;color:#fff;}
.badge.red{background:#a11f2c;color:#fff;}
</style>
"""
st.markdown(HIDE_DEFAULT, unsafe_allow_html=True)

def risk_badge_class(x):
    return "red" if x>=70 else ("orange" if x>=40 else "green")

def reason_badges(reason_text: str):
    parts = [p.strip() for p in (reason_text or "").split(";") if p.strip()]
    html = "".join([f"<span class='badge {'red' if ('High' in p or 'Large' in p) else 'green'}'>{p}</span>" for p in parts])
    return html or "<span class='badge green'>Stable</span>"

# -------- Data config --------
BASE_URL_CG = "https://api.coingecko.com/api/v3"
CACHE_DIR   = pathlib.Path("./cache");   CACHE_DIR.mkdir(exist_ok=True)
OFFLINE_DIR = pathlib.Path("./offline"); OFFLINE_DIR.mkdir(exist_ok=True)
OFFLINE_FILE = OFFLINE_DIR / "markets_sample.json"
TIMEOUT, REQUEST_DELAY, CACHE_TTL = 8, 2.0, 600

# -------- Small utils --------
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

# -------- Primary source (CoinGecko) --------
def fetch_cg_markets(max_coins=30, currency="usd", ttl=CACHE_TTL):
    key=f"cg_{currency}_{max_coins}_spark"
    cached = _read_cache(key, ttl=ttl)
    if cached is not None:
        return cached, "CACHE(CG)"
    url=(f"{BASE_URL_CG}/coins/markets?vs_currency={currency}"
         f"&order=market_cap_desc&per_page={max_coins}&page=1"
         f"&sparkline=true&price_change_percentage=1h,24h,7d,30d")
    r = requests.get(url, timeout=TIMEOUT, headers={"Accept":"application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"CG {r.status_code}")
    data = r.json()
    _write_cache(key, data)
    time.sleep(REQUEST_DELAY)
    return data, "LIVE(CG)"

def norm_from_cg(row):
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "symbol": (row.get("symbol") or "").upper(),
        "current_price": row.get("current_price"),
        "market_cap": row.get("market_cap"),
        "volume_24h": row.get("total_volume"),
        "price_change_percentage_1h":  row.get("price_change_percentage_1h_in_currency", row.get("price_change_percentage_1h")),
        "price_change_percentage_24h": row.get("price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h")),
        "price_change_percentage_7d":  row.get("price_change_percentage_7d_in_currency", row.get("price_change_percentage_7d")),
        "price_change_percentage_30d": row.get("price_change_percentage_30d_in_currency", row.get("price_change_percentage_30d")),
        "sparkline": (row.get("sparkline") or {}).get("price", []),
    }

def load_offline_json():
    if OFFLINE_FILE.exists():
        try: return json.loads(OFFLINE_FILE.read_text())
        except: return []
    return []

def get_markets_with_fallback(source="auto", max_coins=30, currency="usd"):
    """
    Returns: df, src_used, last_error
    """
    last_err = None
    if source in ("auto","primary"):
        try:
            rows, src = fetch_cg_markets(max_coins, currency)
            norm = [norm_from_cg(r) for r in rows][:max_coins]
            # refresh offline snapshot
            try: OFFLINE_FILE.write_text(json.dumps(norm, indent=2))
            except: pass
            df = pd.DataFrame(norm)
            # compute 24/48/72 via sparkline
            df["price_change_percentage_48h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p,48))
            df["price_change_percentage_72h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p,72))
            df["price_change_percentage_24h"] = df.apply(
                lambda r: r["price_change_percentage_24h"] if pd.notna(r["price_change_percentage_24h"])
                else pct_change_from_sparkline(r["sparkline"],24), axis=1)
            return df, src, None
        except Exception as e:
            last_err = e
    raw = load_offline_json()
    if raw:
        df = pd.DataFrame(raw).head(max_coins)
        if "sparkline" in df.columns:
            df["price_change_percentage_48h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p,48))
            df["price_change_percentage_72h"] = df["sparkline"].apply(lambda p: pct_change_from_sparkline(p,72))
        return df, "OFFLINE(JSON)", last_err
    raise RuntimeError(f"No data sources available. Last live error: {last_err}")

# -------- Features + risk --------
def _coerce_numeric(df):
    for c in ["current_price","market_cap","volume_24h",
              "price_change_percentage_1h","price_change_percentage_24h",
              "price_change_percentage_48h","price_change_percentage_72h",
              "price_change_percentage_7d","price_change_percentage_30d"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _norm(s: pd.Series) -> pd.Series:
    s2 = s.replace([np.inf,-np.inf], np.nan)
    m, sd = s2.mean(), s2.std()
    if not np.isfinite(sd) or sd == 0: return pd.Series(0, index=s.index)
    z = (s2 - m) / sd
    return 1/(1+np.exp(-z))

def build_features_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = _coerce_numeric(df.copy())
    df["abs_24h"]      = df["price_change_percentage_24h"].abs()
    df["down_1h"]      = df["price_change_percentage_1h"].clip(upper=0).abs()
    df["volume_ratio"] = (df["volume_24h"]/df["market_cap"]).clip(lower=0, upper=1)
    df["volatility_proxy"] = (
        df["price_change_percentage_1h"].abs().fillna(0) +
        df["price_change_percentage_24h"].abs().fillna(0) +
        df["price_change_percentage_7d"].abs().fillna(0)
    )/3.0
    risk = (0.50*_norm(df["volatility_proxy"]) +
            0.30*_norm(df["abs_24h"]) +
            0.15*_norm(df["down_1h"]) +
            0.05*_norm(df["volume_ratio"]))
    df["risk_score"] = (100*risk).clip(0,100).round(1)
    # reasons
    reasons=[]
    for _, r in df.iterrows():
        why=[]
        if r["volatility_proxy"] > df["volatility_proxy"].median(): why.append("High recent volatility")
        if r["abs_24h"] > df["abs_24h"].median(): why.append("Large 24h move")
        if r["price_change_percentage_1h"] < -0.5: why.append("1h downside")
        if r["volume_ratio"] > 0.5: why.append("Heavy volume vs mcap")
        reasons.append("; ".join(why) if why else "Stable / normal range")
    df["risk_reason"] = reasons
    return df

# -------- Sidebar --------
with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Data source", ["auto","primary","offline"], index=0)
    max_coins = st.slider("Coins", 10, 100, 30)
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
        import pathlib
        for p in pathlib.Path("cache").glob("*.json"): p.unlink()
        st.success("Cache cleared.")

@st.cache_data(ttl=600, show_spinner=False)
def load_and_score(src, n):
    df, src_used, last_err = get_markets_with_fallback(source=src, max_coins=n)
    df = build_features_and_risk(df).sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df, src_used, last_err

# -------- Main --------
st.title("💥 Crashcaster — Early Warning for Crypto Crashes")
df, src_used, last_err = load_and_score(source, max_coins)

# KPIs
col1, col2, col3, col4 = st.columns(4)
if not df.empty:
    top = df.iloc[0]
    col1.metric("Top Risk Coin", f"{top['symbol']}", f"{top['risk_score']:.0f}")
    col2.metric("Market Risk Avg", f"{df['risk_score'].mean():.0f}%")
    col3.metric("Coins ≥ Threshold", f"{(df['risk_score']>=threshold).sum()}/{len(df)}")
    col4.metric("Data Source", src_used)

if src_used.startswith("OFFLINE") and last_err:
    st.warning(f"Using offline snapshot. Live fetch failed with: {last_err}")

# -------- Hero “Try it now” panel --------
st.markdown("### Try it now")
left, right = st.columns([3,1])
with left:
    symbols = df["symbol"].dropna().unique().tolist()
    symbols.sort()
    selected_symbol = st.selectbox("Pick a coin", symbols, index=0)
with right:
    analyze_clicked = st.button("Analyze", use_container_width=True)

if analyze_clicked:
    st.session_state["selected_symbol"] = selected_symbol

if "selected_symbol" in st.session_state:
    sel = st.session_state["selected_symbol"]
    row = df[df["symbol"] == sel].head(1)
    if not row.empty:
        r = row.iloc[0]
        c1, c2 = st.columns([2,1])
        with c1:
            st.subheader(f"{r['name']} ({r['symbol']})")
            spark = r.get("sparkline", None)
            if isinstance(spark, (list, tuple)) and len(spark) > 5:
                fig = go.Figure(go.Scatter(y=spark, mode="lines"))
                fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            s24 = r.get("price_change_percentage_24h", 0) or 0
            s48 = r.get("price_change_percentage_48h", np.nan)
            s72 = r.get("price_change_percentage_72h", np.nan)
            st.markdown(
                f"**24h:** {s24:+.2f}% · **48h:** {s48 if pd.notna(s48) else '—'}% · "
                f"**72h:** {s72 if pd.notna(s72) else '—'}%"
            )
        with c2:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(r["risk_score"]),
                                           gauge={"axis":{"range":[0,100]}}, title={"text":"Risk score"}))
            gauge.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(f"<div>{reason_badges(r['risk_reason'])}</div>", unsafe_allow_html=True)

# -------- Tabs --------
tab1, tab2 = st.tabs(["📊 Dashboard","✅ Recommendations"])

with tab1:
    st.subheader("At-Risk Coins")
    danger = df[df["risk_score"]>=threshold]
    st.dataframe(danger[[
        "name","symbol","current_price",
        "price_change_percentage_24h","price_change_percentage_48h","price_change_percentage_72h",
        "price_change_percentage_7d",
        "volatility_proxy","volume_24h","market_cap","risk_score","risk_reason"
    ]], use_container_width=True)

    st.subheader("All Coins")
    st.dataframe(df[[
        "name","symbol","current_price",
        "price_change_percentage_24h","price_change_percentage_48h","price_change_percentage_72h",
        "price_change_percentage_7d",
        "volatility_proxy","volume_24h","market_cap","risk_score","risk_reason"
    ]], use_container_width=True)

    st.subheader("Charts")
    top_risk = df.nlargest(10, "risk_score")
    fig1 = px.bar(top_risk, x="symbol", y="risk_score", title="Top-10 Crash Risk")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_bar(name="24h %", x=df["symbol"], y=df["price_change_percentage_24h"])
    fig2.add_bar(name="48h %", x=df["symbol"], y=df["price_change_percentage_48h"])
    fig2.add_bar(name="72h %", x=df["symbol"], y=df["price_change_percentage_72h"])
    fig2.update_layout(barmode="group", title="Price Change: 24/48/72h")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    def recommend_coins_local(dff, strategy="trend", top_k=10):
        d = dff.copy()
        for col in ["price_change_percentage_24h","price_change_percentage_48h",
                    "price_change_percentage_72h","volatility_proxy","risk_score"]:
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
            score = (0.60*c24 + 0.25*c48 + 0.15*c72 - 0.20*vol)
        else:  # reversal
            mask = ((d["risk_score"] < 55) & (c24 < 0) &
                    ok_nonneg_or_nan(d["price_change_percentage_72h"]) &
                    (vol < vol.quantile(0.8)))
            score = ((-1.0)*c24 + 0.30*c72 - 0.20*vol)
        out = d[mask].copy()
        out["recommend_score"] = score[mask]
        out = out.sort_values("recommend_score", ascending=False).head(top_k)
        if out.empty:
            fb = d[(d["risk_score"] < 60)].copy()
            fb["recommend_score"] = 0.70*c24 - 0.20*vol
            out = fb.sort_values("recommend_score", ascending=False).head(top_k)
            st.info("Showing fallback list (limited 48/72h data). Refresh the offline snapshot when online to improve results.")
        cols = ["name","symbol","current_price",
                "price_change_percentage_24h","price_change_percentage_48h","price_change_percentage_72h",
                "volatility_proxy","risk_score","risk_reason","recommend_score"]
        return out[cols]

    strat = st.selectbox("Strategy", ["trend","reversal"], index=0)
    rec = recommend_coins_local(df, strat, top_k=10)
    st.dataframe(rec, use_container_width=True)

st.caption("Tip: If the API hiccups, create an offline snapshot (sidebar) and set Data source → 'offline'. Not financial advice.")
