
# Crashcaster — Project Documentation

## Overview
Crashcaster is a Streamlit web application that provides an **early-warning dashboard for potential crypto-currency crashes**.  
It fetches market data (primarily from CoinGecko), engineers several volatility / liquidity features, calculates an aggregate *risk score* per coin, and visualises the results with interactive components.

This document is generated automatically and is intended to serve as a **single source of truth** for all public APIs, utility functions and UI components available in the code-base.

---

## Quick-Start

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

The application will open in your default browser on http://localhost:8501.  Adjust settings in the **sidebar** and explore the *Try it now* panel as well as the *Dashboard* and *Recommendations* tabs.

---

## API Reference (Python)
All public (non-underscore-prefixed) helpers live in `app.py`.  Importing from the module allows you to reuse the data pipeline outside the Streamlit context—e.g. in notebooks, scheduled jobs, or unit tests.

```python
import app as crashcaster
```

> **Tip**   Each function returns *pure* data structures (lists / pandas DataFrames) and can therefore be composed freely.

### 1. Data Fetching
#### `fetch_cg_markets(max_coins: int = 30, currency: str = "usd", ttl: int = 600) -> tuple[list[dict], str]`
Fetches live market data from the CoinGecko `coins/markets` endpoint and transparently caches responses on disk (🔄 `./cache/<key>.json`).

Parameters:
• **max_coins** – Upper limit of coins returned, ordered by market-cap.  
• **currency** – Quote currency for price fields (`usd`, `eur`, …).  
• **ttl** – Cache time-to-live (seconds).

Returns `(data, source)` where `data` is a list of raw CoinGecko records and `source` is either `"LIVE(CG)"` or `"CACHE(CG)"`.

Example:
```python
rows, src = crashcaster.fetch_cg_markets(max_coins=50)
print(f"Fetched {len(rows)} coins from {src}")
```

#### `norm_from_cg(row: dict) -> dict`
Normalises a single CoinGecko record to the project’s canonical schema.

Example:
```python
first = rows[0]
clean = crashcaster.norm_from_cg(first)
print(clean.keys())
```

#### `get_markets_with_fallback(source: str = "auto", max_coins: int = 30, currency: str = "usd") -> tuple[pd.DataFrame, str, Exception | None]`
High-level helper that tries **live fetch ➜ offline snapshot** in one call.

• **source** – `"auto"` → live first, fall back to snapshot;  
  `"primary"` → live only;  
  `"offline"` → snapshot only.

Returns a DataFrame (normalised), the data‐source indicator string, and the last live error (if any).

Example:
```python
df, src, err = crashcaster.get_markets_with_fallback()
print(df.head())
```

### 2. Feature Engineering & Risk
#### `build_features_and_risk(df: pd.DataFrame) -> pd.DataFrame`
Adds engineered features (volatility proxy, volume ratio, …) **and** a final `risk_score` column scaled 0-100.  Also attaches a human-readable `risk_reason`.

Example:
```python
df_scored = crashcaster.build_features_and_risk(df)
print(df_scored[["symbol", "risk_score"]].head())
```

### 3. Utility Helpers
| Function | Purpose |
|----------|---------|
| `risk_badge_class(x: float) -> str` | Map a numeric risk score to a CSS colour class (`green`, `orange`, `red`). |
| `reason_badges(reason_text: str) -> str` | Convert semicolon-separated reasons to pre-styled `<span>` badges for inline HTML use. |
| `pct_change_from_sparkline(prices: list[float], hours: int) -> float | None` | Percentage change between two points in the mini price history vector. |

---

## Streamlit Components
While functions above can be reused standalone, the following Streamlit elements constitute the *web interface*:

1. **Sidebar** – data source selector, coin limit slider, risk threshold slider, utilities buttons.
2. **Hero "Try it now" panel** – coin picker with instant analytics & gauge.
3. **Tabs**
   • *Dashboard* – at-risk table, full table, bar & grouped bar charts.  
   • *Recommendations* – alpha strategies (*trend* / *reversal*) with top-10 suggestions.

Developers can use these blocks as reference for embedding the computational helpers into their own Streamlit or Dash apps.

---

## Working Offline
1. Launch the app once **online** and click *Refresh offline snapshot* in the sidebar.  
   A file `./offline/markets_sample.json` will be created.
2. Later, on a machine without internet access, pick *Data source → offline*.  
   All computations / charts will operate on the saved snapshot.

---

## Changelog
*2024-08-15* – Initial comprehensive documentation generated automatically.

---

## License
Distributed under the MIT License.  See `LICENSE` for more information.