# Crashcaster API Documentation

## Overview

Crashcaster is a Streamlit-based early warning system for cryptocurrency crashes. It fetches market data from CoinGecko API, calculates risk scores using various financial indicators, and provides an interactive dashboard for monitoring cryptocurrency market conditions.

## Table of Contents

1. [Data Fetching & Caching](#data-fetching--caching)
2. [Data Processing & Normalization](#data-processing--normalization)
3. [Risk Scoring Engine](#risk-scoring-engine)
4. [UI Components & Styling](#ui-components--styling)
5. [Utility Functions](#utility-functions)
6. [Configuration](#configuration)

---

## Data Fetching & Caching

### `fetch_cg_markets(max_coins=30, currency="usd", ttl=CACHE_TTL)`

Fetches cryptocurrency market data from the CoinGecko API with caching support.

**Parameters:**
- `max_coins` (int, default=30): Maximum number of coins to fetch
- `currency` (str, default="usd"): Base currency for price data
- `ttl` (int, default=600): Cache time-to-live in seconds

**Returns:**
- `tuple`: (data, source_indicator)
  - `data` (list): Raw market data from CoinGecko API
  - `source_indicator` (str): Either "CACHE(CG)" or "LIVE(CG)"

**Raises:**
- `RuntimeError`: If API request fails (non-200 status code)

**Example:**
```python
# Fetch top 50 coins in EUR
data, source = fetch_cg_markets(max_coins=50, currency="eur")
print(f"Fetched {len(data)} coins from {source}")
```

### `get_markets_with_fallback(source="auto", max_coins=30, currency="usd")`

Primary data fetching function with automatic fallback to offline data.

**Parameters:**
- `source` (str, default="auto"): Data source strategy
  - `"auto"`: Try live API first, fallback to offline
  - `"primary"`: Only use live API
  - `"offline"`: Only use cached offline data
- `max_coins` (int, default=30): Maximum number of coins to return
- `currency` (str, default="usd"): Base currency for price data

**Returns:**
- `tuple`: (dataframe, source_used, last_error)
  - `dataframe` (pd.DataFrame): Processed market data
  - `source_used` (str): Actual data source used
  - `last_error` (Exception or None): Last error encountered

**Example:**
```python
# Get market data with automatic fallback
df, source, error = get_markets_with_fallback(source="auto", max_coins=100)
if error:
    print(f"Warning: {error}")
print(f"Loaded {len(df)} coins from {source}")
```

### Cache Management Functions

#### `_read_cache(key, ttl=CACHE_TTL)`

Reads cached data from disk if it exists and is not expired.

**Parameters:**
- `key` (str): Cache key identifier
- `ttl` (int, default=600): Time-to-live in seconds

**Returns:**
- `dict or None`: Cached data if valid, None otherwise

#### `_write_cache(key, data)`

Writes data to disk cache.

**Parameters:**
- `key` (str): Cache key identifier
- `data` (dict): Data to cache

#### `load_offline_json()`

Loads offline market data snapshot from disk.

**Returns:**
- `list`: Offline market data or empty list if not available

---

## Data Processing & Normalization

### `norm_from_cg(row)`

Normalizes raw CoinGecko API response into standardized format.

**Parameters:**
- `row` (dict): Raw market data row from CoinGecko API

**Returns:**
- `dict`: Normalized market data with standardized field names

**Normalized Fields:**
- `id`: CoinGecko coin identifier
- `name`: Full coin name
- `symbol`: Trading symbol (uppercase)
- `current_price`: Current price in base currency
- `market_cap`: Market capitalization
- `volume_24h`: 24-hour trading volume
- `price_change_percentage_1h/24h/7d/30d`: Price change percentages
- `sparkline`: Historical price data array

**Example:**
```python
# Normalize CoinGecko data
raw_data = {"id": "bitcoin", "symbol": "btc", "current_price": 45000}
normalized = norm_from_cg(raw_data)
print(normalized["symbol"])  # "BTC"
```

### `pct_change_from_sparkline(prices, hours)`

Calculates percentage change from sparkline price data.

**Parameters:**
- `prices` (list): Array of historical prices
- `hours` (int): Number of hours to look back

**Returns:**
- `float or None`: Percentage change, or None if insufficient data

**Example:**
```python
# Calculate 48-hour price change
prices = [100, 102, 98, 105, 103]
change = pct_change_from_sparkline(prices, 2)
print(f"48h change: {change:.2f}%")
```

---

## Risk Scoring Engine

### `build_features_and_risk(df)`

Main risk scoring function that calculates risk scores and reasons for each cryptocurrency.

**Parameters:**
- `df` (pd.DataFrame): Market data with price and volume information

**Returns:**
- `pd.DataFrame`: Enhanced dataframe with risk features and scores

**Added Columns:**
- `abs_24h`: Absolute 24-hour price change
- `down_1h`: Downward 1-hour price movement (clipped to negative values)
- `volume_ratio`: Trading volume to market cap ratio
- `volatility_proxy`: Composite volatility measure
- `risk_score`: Final risk score (0-100)
- `risk_reason`: Human-readable explanation of risk factors

**Risk Scoring Algorithm:**
```
risk_score = (
    50% * normalized(volatility_proxy) +
    30% * normalized(abs_24h) +
    15% * normalized(down_1h) +
    5% * normalized(volume_ratio)
) * 100
```

**Example:**
```python
# Calculate risk scores for market data
df_with_risk = build_features_and_risk(market_df)
high_risk = df_with_risk[df_with_risk['risk_score'] >= 70]
print(f"Found {len(high_risk)} high-risk coins")
```

### Risk Feature Functions

#### `_coerce_numeric(df)`

Converts relevant columns to numeric types, handling errors gracefully.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with numeric columns converted

#### `_norm(s)`

Normalizes a pandas Series using sigmoid transformation of z-scores.

**Parameters:**
- `s` (pd.Series): Input series to normalize

**Returns:**
- `pd.Series`: Normalized values between 0 and 1

---

## UI Components & Styling

### `risk_badge_class(x)`

Determines CSS class for risk score badges based on value.

**Parameters:**
- `x` (float): Risk score value

**Returns:**
- `str`: CSS class name ("red", "orange", or "green")

**Thresholds:**
- `>= 70`: "red" (high risk)
- `>= 40`: "orange" (medium risk)
- `< 40`: "green" (low risk)

**Example:**
```python
risk_score = 75
badge_class = risk_badge_class(risk_score)
print(f"Risk {risk_score} gets {badge_class} badge")  # "red"
```

### `reason_badges(reason_text)`

Converts risk reason text into HTML badge elements.

**Parameters:**
- `reason_text` (str): Semicolon-separated risk factors

**Returns:**
- `str`: HTML string with styled badge elements

**Example:**
```python
reasons = "High recent volatility; Large 24h move"
html = reason_badges(reasons)
# Returns: "<span class='badge red'>High recent volatility</span><span class='badge red'>Large 24h move</span>"
```

### Streamlit Components

#### Main Dashboard Components

1. **KPI Metrics Row**: Displays top risk coin, average market risk, coins above threshold, and data source
2. **Interactive Coin Analyzer**: Dropdown selection with detailed analysis including sparkline charts and risk gauges
3. **Risk Dashboard Tab**: Tables showing at-risk coins and comprehensive market data
4. **Recommendations Tab**: Algorithmic coin recommendations based on trend or reversal strategies

#### Sidebar Controls

- **Data Source Selector**: Choose between auto, primary (live), or offline data
- **Coin Count Slider**: Adjust number of coins to analyze (10-100)
- **Risk Threshold Slider**: Set risk score threshold for alerts (0-100)
- **Utility Buttons**: Refresh offline snapshot and clear API cache

---

## Utility Functions

### `_cache_path(key)`

Generates file path for cache storage.

**Parameters:**
- `key` (str): Cache key

**Returns:**
- `pathlib.Path`: Full path to cache file

### Streamlit Cached Functions

#### `load_and_score(src, n)`

Streamlit cached function that loads market data and calculates risk scores.

**Parameters:**
- `src` (str): Data source ("auto", "primary", or "offline")
- `n` (int): Number of coins to load

**Returns:**
- `tuple`: (dataframe, source_used, last_error)

**Caching:**
- TTL: 600 seconds (10 minutes)
- Automatically invalidated when parameters change

#### `recommend_coins_local(dff, strategy="trend", top_k=10)`

Generates coin recommendations based on specified strategy.

**Parameters:**
- `dff` (pd.DataFrame): Market data with risk scores
- `strategy` (str): Recommendation strategy
  - `"trend"`: Coins with positive momentum and low risk
  - `"reversal"`: Oversold coins with recovery potential
- `top_k` (int, default=10): Number of recommendations to return

**Returns:**
- `pd.DataFrame`: Top recommended coins with recommendation scores

**Strategy Details:**

**Trend Strategy:**
- Filters: risk_score < 65, low volatility, positive 24h change
- Score: `0.60*change_24h + 0.25*change_48h + 0.15*change_72h - 0.20*volatility`

**Reversal Strategy:**
- Filters: risk_score < 55, negative 24h change, low volatility
- Score: `(-1.0)*change_24h + 0.30*change_72h - 0.20*volatility`

**Example:**
```python
# Get trend-based recommendations
recommendations = recommend_coins_local(df, strategy="trend", top_k=5)
print(f"Top recommendation: {recommendations.iloc[0]['symbol']}")
```

---

## Configuration

### Constants

```python
# API Configuration
BASE_URL_CG = "https://api.coingecko.com/api/v3"
TIMEOUT = 8  # Request timeout in seconds
REQUEST_DELAY = 2.0  # Delay between requests
CACHE_TTL = 600  # Default cache time-to-live

# Directory Structure
CACHE_DIR = "./cache"  # API response cache
OFFLINE_DIR = "./offline"  # Offline data storage
OFFLINE_FILE = "./offline/markets_sample.json"  # Offline data file
```

### Streamlit Configuration

The app uses a dark theme configuration in `.streamlit/config.toml`:

```toml
[theme]
base="dark"
primaryColor="#FF4B4B"
backgroundColor="#0E1117"
secondaryBackgroundColor="#262730"
textColor="#FAFAFA"
font="sans serif"
```

### Page Configuration

```python
st.set_page_config(
    page_title="Crashcaster — Early Warning",
    page_icon="💥",
    layout="wide"
)
```

---

## Error Handling

The application implements robust error handling:

1. **API Failures**: Automatic fallback to offline data when live API fails
2. **Data Validation**: Numeric coercion with error handling for malformed data
3. **Cache Corruption**: Graceful handling of corrupted cache files
4. **Missing Data**: Fallback values and indicators for missing sparkline data

## Performance Optimizations

1. **Caching**: Multi-level caching (Streamlit cache + disk cache)
2. **Data Limits**: Configurable limits on data fetching to manage API quotas
3. **Request Throttling**: Built-in delays to respect API rate limits
4. **Offline Mode**: Complete offline functionality for development and testing