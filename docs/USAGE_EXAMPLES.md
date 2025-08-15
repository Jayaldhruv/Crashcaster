# Crashcaster Usage Examples & Tutorials

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Data Fetching Examples](#data-fetching-examples)
3. [Risk Analysis Examples](#risk-analysis-examples)
4. [Custom Dashboard Components](#custom-dashboard-components)
5. [Advanced Usage Patterns](#advanced-usage-patterns)
6. [Integration Examples](#integration-examples)

---

## Quick Start Guide

### Basic Application Setup

```python
import streamlit as st
import pandas as pd
import numpy as np
from app import get_markets_with_fallback, build_features_and_risk

# Initialize Streamlit app
st.set_page_config(
    page_title="My Crypto Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load market data
df, source, error = get_markets_with_fallback(source="auto", max_coins=50)

# Calculate risk scores
df_with_risk = build_features_and_risk(df)

# Display results
st.title("Cryptocurrency Risk Monitor")
st.dataframe(df_with_risk[['name', 'symbol', 'risk_score', 'risk_reason']])
```

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Access the app at http://localhost:8501
```

---

## Data Fetching Examples

### Example 1: Basic Data Fetching

```python
from app import fetch_cg_markets, norm_from_cg

# Fetch top 20 coins in USD
try:
    raw_data, source = fetch_cg_markets(max_coins=20, currency="usd")
    print(f"Fetched {len(raw_data)} coins from {source}")
    
    # Normalize the first coin's data
    if raw_data:
        normalized = norm_from_cg(raw_data[0])
        print(f"Bitcoin price: ${normalized['current_price']:,.2f}")
        print(f"24h change: {normalized['price_change_percentage_24h']:.2f}%")
        
except Exception as e:
    print(f"Error fetching data: {e}")
```

### Example 2: Multi-Currency Data Fetching

```python
from app import fetch_cg_markets
import pandas as pd

currencies = ["usd", "eur", "btc"]
all_data = {}

for currency in currencies:
    try:
        data, source = fetch_cg_markets(max_coins=10, currency=currency)
        all_data[currency] = data
        print(f"Fetched {currency.upper()} data from {source}")
    except Exception as e:
        print(f"Failed to fetch {currency} data: {e}")

# Compare Bitcoin prices across currencies
if all_data:
    for currency, data in all_data.items():
        btc_data = next((coin for coin in data if coin['id'] == 'bitcoin'), None)
        if btc_data:
            price = btc_data.get('current_price', 0)
            print(f"BTC price in {currency.upper()}: {price:,.8f}")
```

### Example 3: Offline Data Management

```python
from app import load_offline_json, get_markets_with_fallback
import json

# Create offline backup
def create_offline_backup():
    try:
        df, source, error = get_markets_with_fallback(source="primary", max_coins=100)
        
        # Convert DataFrame to JSON-serializable format
        backup_data = df.to_dict('records')
        
        # Save to offline file
        with open('./offline/markets_backup.json', 'w') as f:
            json.dump(backup_data, f, indent=2)
            
        print(f"Created offline backup with {len(backup_data)} coins")
        
    except Exception as e:
        print(f"Backup creation failed: {e}")

# Load offline data
def load_offline_backup():
    try:
        offline_data = load_offline_json()
        if offline_data:
            df = pd.DataFrame(offline_data)
            print(f"Loaded {len(df)} coins from offline storage")
            return df
        else:
            print("No offline data available")
            return pd.DataFrame()
    except Exception as e:
        print(f"Failed to load offline data: {e}")
        return pd.DataFrame()

# Usage
create_offline_backup()
offline_df = load_offline_backup()
```

---

## Risk Analysis Examples

### Example 1: Basic Risk Scoring

```python
from app import get_markets_with_fallback, build_features_and_risk
import pandas as pd

# Get market data and calculate risk scores
df, source, error = get_markets_with_fallback(max_coins=50)
df_risk = build_features_and_risk(df)

# Analyze high-risk coins
high_risk = df_risk[df_risk['risk_score'] >= 70]
print(f"Found {len(high_risk)} high-risk coins:")

for _, coin in high_risk.iterrows():
    print(f"- {coin['name']} ({coin['symbol']}): {coin['risk_score']:.1f}")
    print(f"  Reason: {coin['risk_reason']}")
```

### Example 2: Custom Risk Thresholds

```python
from app import build_features_and_risk

def analyze_risk_distribution(df, thresholds=[40, 70, 85]):
    """Analyze risk distribution across different thresholds."""
    
    df_risk = build_features_and_risk(df)
    
    risk_categories = {
        'Low Risk': df_risk[df_risk['risk_score'] < thresholds[0]],
        'Medium Risk': df_risk[
            (df_risk['risk_score'] >= thresholds[0]) & 
            (df_risk['risk_score'] < thresholds[1])
        ],
        'High Risk': df_risk[
            (df_risk['risk_score'] >= thresholds[1]) & 
            (df_risk['risk_score'] < thresholds[2])
        ],
        'Critical Risk': df_risk[df_risk['risk_score'] >= thresholds[2]]
    }
    
    print("Risk Distribution Analysis:")
    print("-" * 40)
    
    for category, coins in risk_categories.items():
        print(f"{category}: {len(coins)} coins")
        if not coins.empty:
            avg_risk = coins['risk_score'].mean()
            print(f"  Average risk score: {avg_risk:.1f}")
            top_coin = coins.nlargest(1, 'risk_score')
            if not top_coin.empty:
                print(f"  Highest: {top_coin.iloc[0]['symbol']} ({top_coin.iloc[0]['risk_score']:.1f})")
        print()

# Usage
df, _, _ = get_markets_with_fallback(max_coins=100)
analyze_risk_distribution(df, thresholds=[30, 60, 80])
```

### Example 3: Risk Trend Analysis

```python
from app import pct_change_from_sparkline, build_features_and_risk
import pandas as pd

def analyze_price_trends(df):
    """Analyze price trends using sparkline data."""
    
    df_analysis = df.copy()
    
    # Calculate additional timeframe changes
    timeframes = [6, 12, 24, 48, 72, 168]  # hours
    
    for hours in timeframes:
        col_name = f'change_{hours}h'
        df_analysis[col_name] = df_analysis['sparkline'].apply(
            lambda prices: pct_change_from_sparkline(prices, hours)
        )
    
    # Identify trend patterns
    def classify_trend(row):
        changes = [row.get(f'change_{h}h', 0) for h in [24, 48, 72]]
        changes = [c for c in changes if pd.notna(c)]
        
        if not changes:
            return "Unknown"
        
        if all(c > 2 for c in changes):
            return "Strong Uptrend"
        elif all(c > 0 for c in changes):
            return "Uptrend"
        elif all(c < -2 for c in changes):
            return "Strong Downtrend"
        elif all(c < 0 for c in changes):
            return "Downtrend"
        else:
            return "Sideways"
    
    df_analysis['trend_pattern'] = df_analysis.apply(classify_trend, axis=1)
    
    # Add risk scores
    df_risk = build_features_and_risk(df_analysis)
    
    return df_risk

# Usage example
df, _, _ = get_markets_with_fallback(max_coins=30)
trend_analysis = analyze_price_trends(df)

# Show coins by trend pattern
for pattern in trend_analysis['trend_pattern'].unique():
    coins = trend_analysis[trend_analysis['trend_pattern'] == pattern]
    avg_risk = coins['risk_score'].mean()
    print(f"{pattern}: {len(coins)} coins, avg risk: {avg_risk:.1f}")
```

---

## Custom Dashboard Components

### Example 1: Custom Risk Gauge

```python
import streamlit as st
import plotly.graph_objects as go

def create_risk_gauge(risk_score, title="Risk Score"):
    """Create a custom risk gauge component."""
    
    # Determine color based on risk score
    if risk_score >= 70:
        color = "red"
    elif risk_score >= 40:
        color = "orange"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Usage in Streamlit
st.title("Custom Risk Dashboard")

# Sample data
risk_scores = [25, 55, 85]
coin_names = ["Bitcoin", "Ethereum", "Dogecoin"]

cols = st.columns(3)
for i, (score, name) in enumerate(zip(risk_scores, coin_names)):
    with cols[i]:
        fig = create_risk_gauge(score, f"{name} Risk")
        st.plotly_chart(fig, use_container_width=True)
```

### Example 2: Interactive Risk Matrix

```python
import streamlit as st
import plotly.express as px
import pandas as pd

def create_risk_matrix(df):
    """Create an interactive risk vs. volatility matrix."""
    
    # Ensure we have the required columns
    if 'volatility_proxy' not in df.columns or 'risk_score' not in df.columns:
        st.error("DataFrame must contain 'volatility_proxy' and 'risk_score' columns")
        return None
    
    # Create the scatter plot
    fig = px.scatter(
        df,
        x='volatility_proxy',
        y='risk_score',
        size='market_cap',
        color='price_change_percentage_24h',
        hover_name='name',
        hover_data=['symbol', 'current_price'],
        title="Risk vs. Volatility Matrix",
        labels={
            'volatility_proxy': 'Volatility Proxy',
            'risk_score': 'Risk Score',
            'price_change_percentage_24h': '24h Change (%)'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.add_vline(x=df['volatility_proxy'].median(), line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig.add_annotation(x=df['volatility_proxy'].max() * 0.8, y=80, 
                      text="High Risk<br>High Volatility", showarrow=False)
    fig.add_annotation(x=df['volatility_proxy'].max() * 0.2, y=80, 
                      text="High Risk<br>Low Volatility", showarrow=False)
    fig.add_annotation(x=df['volatility_proxy'].max() * 0.8, y=20, 
                      text="Low Risk<br>High Volatility", showarrow=False)
    fig.add_annotation(x=df['volatility_proxy'].max() * 0.2, y=20, 
                      text="Low Risk<br>Low Volatility", showarrow=False)
    
    return fig

# Usage
df, _, _ = get_markets_with_fallback(max_coins=50)
df_risk = build_features_and_risk(df)

st.title("Risk Analysis Dashboard")
matrix_fig = create_risk_matrix(df_risk)
if matrix_fig:
    st.plotly_chart(matrix_fig, use_container_width=True)
```

### Example 3: Real-time Alert System

```python
import streamlit as st
import time
from datetime import datetime

def create_alert_system(risk_threshold=70, check_interval=60):
    """Create a real-time alert system for high-risk coins."""
    
    # Initialize session state for alerts
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'last_check' not in st.session_state:
        st.session_state.last_check = datetime.now()
    
    # Create alert container
    alert_container = st.container()
    
    # Auto-refresh every minute
    if st.button("Check for Alerts") or (datetime.now() - st.session_state.last_check).seconds > check_interval:
        
        with st.spinner("Checking for high-risk coins..."):
            df, source, error = get_markets_with_fallback(max_coins=100)
            df_risk = build_features_and_risk(df)
            
            # Find high-risk coins
            high_risk_coins = df_risk[df_risk['risk_score'] >= risk_threshold]
            
            # Generate alerts
            current_time = datetime.now().strftime("%H:%M:%S")
            for _, coin in high_risk_coins.iterrows():
                alert_msg = {
                    'time': current_time,
                    'coin': f"{coin['name']} ({coin['symbol']})",
                    'risk_score': coin['risk_score'],
                    'reason': coin['risk_reason'],
                    'price': coin['current_price']
                }
                
                # Only add if not already in recent alerts
                if not any(alert['coin'] == alert_msg['coin'] for alert in st.session_state.alerts[-10:]):
                    st.session_state.alerts.append(alert_msg)
            
            st.session_state.last_check = datetime.now()
    
    # Display alerts
    with alert_container:
        st.subheader("🚨 High-Risk Alerts")
        
        if st.session_state.alerts:
            # Show latest 10 alerts
            recent_alerts = st.session_state.alerts[-10:]
            
            for alert in reversed(recent_alerts):
                with st.expander(f"⚠️ {alert['coin']} - Risk: {alert['risk_score']:.1f}", expanded=False):
                    st.write(f"**Time:** {alert['time']}")
                    st.write(f"**Price:** ${alert['price']:,.4f}")
                    st.write(f"**Risk Factors:** {alert['reason']}")
        else:
            st.info("No high-risk alerts at this time.")
    
    # Clear alerts button
    if st.button("Clear Alerts"):
        st.session_state.alerts = []
        st.experimental_rerun()

# Usage
st.title("Crypto Alert System")
create_alert_system(risk_threshold=75, check_interval=30)
```

---

## Advanced Usage Patterns

### Example 1: Custom Risk Model

```python
from app import _norm
import pandas as pd
import numpy as np

class CustomRiskModel:
    """Custom risk scoring model with configurable weights."""
    
    def __init__(self, weights=None):
        self.weights = weights or {
            'volatility': 0.40,
            'momentum': 0.25,
            'volume': 0.15,
            'trend': 0.10,
            'market_cap': 0.10
        }
    
    def calculate_features(self, df):
        """Calculate custom risk features."""
        df = df.copy()
        
        # Volatility features
        df['volatility_1h'] = df['price_change_percentage_1h'].abs()
        df['volatility_24h'] = df['price_change_percentage_24h'].abs()
        df['volatility_7d'] = df['price_change_percentage_7d'].abs()
        
        # Momentum features
        df['momentum_short'] = df['price_change_percentage_1h']
        df['momentum_medium'] = df['price_change_percentage_24h']
        df['momentum_long'] = df['price_change_percentage_7d']
        
        # Volume features
        df['volume_ratio'] = df['volume_24h'] / df['market_cap']
        df['volume_normalized'] = df['volume_24h'] / df['volume_24h'].median()
        
        # Market cap features
        df['market_cap_rank'] = df['market_cap'].rank(ascending=False)
        df['market_cap_score'] = 1 / (1 + df['market_cap_rank'] / 100)
        
        return df
    
    def calculate_risk_score(self, df):
        """Calculate custom risk scores."""
        df = self.calculate_features(df)
        
        # Normalize features
        volatility_score = _norm(
            df['volatility_1h'] + df['volatility_24h'] + df['volatility_7d']
        )
        
        momentum_score = _norm(
            -df['momentum_short'].fillna(0)  # Negative momentum increases risk
        )
        
        volume_score = _norm(df['volume_ratio'].fillna(0))
        
        trend_score = _norm(
            -df['momentum_medium'].fillna(0) - df['momentum_long'].fillna(0)
        )
        
        market_cap_score = _norm(1 / df['market_cap'].fillna(1))
        
        # Weighted combination
        risk_score = (
            self.weights['volatility'] * volatility_score +
            self.weights['momentum'] * momentum_score +
            self.weights['volume'] * volume_score +
            self.weights['trend'] * trend_score +
            self.weights['market_cap'] * market_cap_score
        ) * 100
        
        df['custom_risk_score'] = risk_score.clip(0, 100)
        
        return df

# Usage example
df, _, _ = get_markets_with_fallback(max_coins=50)

# Create custom risk model
custom_model = CustomRiskModel(weights={
    'volatility': 0.50,
    'momentum': 0.30,
    'volume': 0.10,
    'trend': 0.05,
    'market_cap': 0.05
})

# Calculate custom risk scores
df_custom = custom_model.calculate_risk_score(df)

# Compare with default risk scores
df_default = build_features_and_risk(df)

comparison = pd.DataFrame({
    'coin': df_default['symbol'],
    'default_risk': df_default['risk_score'],
    'custom_risk': df_custom['custom_risk_score']
})

print("Risk Score Comparison:")
print(comparison.head(10))
```

### Example 2: Portfolio Risk Assessment

```python
def assess_portfolio_risk(portfolio, market_data):
    """Assess risk for a cryptocurrency portfolio."""
    
    # Portfolio format: [{'symbol': 'BTC', 'amount': 0.5, 'value_usd': 25000}, ...]
    
    portfolio_df = pd.DataFrame(portfolio)
    
    # Merge with market data
    market_risk = build_features_and_risk(market_data)
    portfolio_risk = portfolio_df.merge(
        market_risk[['symbol', 'risk_score', 'volatility_proxy', 'price_change_percentage_24h']],
        on='symbol',
        how='left'
    )
    
    # Calculate portfolio metrics
    total_value = portfolio_risk['value_usd'].sum()
    portfolio_risk['weight'] = portfolio_risk['value_usd'] / total_value
    portfolio_risk['weighted_risk'] = portfolio_risk['weight'] * portfolio_risk['risk_score']
    
    # Portfolio-level metrics
    portfolio_risk_score = portfolio_risk['weighted_risk'].sum()
    
    # Diversification score (lower is better diversified)
    diversification_score = (portfolio_risk['weight'] ** 2).sum()
    
    # Concentration risk (percentage in top holding)
    concentration_risk = portfolio_risk['weight'].max()
    
    results = {
        'portfolio_risk_score': portfolio_risk_score,
        'diversification_score': diversification_score,
        'concentration_risk': concentration_risk,
        'total_value': total_value,
        'holdings': portfolio_risk.to_dict('records')
    }
    
    return results

# Example usage
sample_portfolio = [
    {'symbol': 'BTC', 'amount': 0.5, 'value_usd': 25000},
    {'symbol': 'ETH', 'amount': 10, 'value_usd': 15000},
    {'symbol': 'ADA', 'amount': 5000, 'value_usd': 2500},
    {'symbol': 'DOT', 'amount': 500, 'value_usd': 2500}
]

df, _, _ = get_markets_with_fallback(max_coins=100)
portfolio_assessment = assess_portfolio_risk(sample_portfolio, df)

print(f"Portfolio Risk Score: {portfolio_assessment['portfolio_risk_score']:.1f}")
print(f"Diversification Score: {portfolio_assessment['diversification_score']:.3f}")
print(f"Concentration Risk: {portfolio_assessment['concentration_risk']:.1%}")
```

---

## Integration Examples

### Example 1: Webhook Notifications

```python
import requests
import json
from datetime import datetime

class WebhookNotifier:
    """Send notifications via webhooks when risk thresholds are exceeded."""
    
    def __init__(self, webhook_urls):
        self.webhook_urls = webhook_urls if isinstance(webhook_urls, list) else [webhook_urls]
    
    def send_risk_alert(self, coin_data, risk_threshold=70):
        """Send alert if coin exceeds risk threshold."""
        
        if coin_data['risk_score'] >= risk_threshold:
            
            message = {
                "text": f"🚨 High Risk Alert: {coin_data['name']} ({coin_data['symbol']})",
                "attachments": [
                    {
                        "color": "danger",
                        "fields": [
                            {"title": "Risk Score", "value": f"{coin_data['risk_score']:.1f}", "short": True},
                            {"title": "Current Price", "value": f"${coin_data['current_price']:,.4f}", "short": True},
                            {"title": "24h Change", "value": f"{coin_data['price_change_percentage_24h']:.2f}%", "short": True},
                            {"title": "Risk Factors", "value": coin_data['risk_reason'], "short": False}
                        ],
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            for webhook_url in self.webhook_urls:
                try:
                    response = requests.post(webhook_url, json=message, timeout=10)
                    response.raise_for_status()
                    print(f"Alert sent successfully to {webhook_url}")
                except Exception as e:
                    print(f"Failed to send alert to {webhook_url}: {e}")

# Usage
webhook_notifier = WebhookNotifier([
    "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK"
])

# Monitor and send alerts
df, _, _ = get_markets_with_fallback(max_coins=50)
df_risk = build_features_and_risk(df)

for _, coin in df_risk.iterrows():
    webhook_notifier.send_risk_alert(coin.to_dict(), risk_threshold=75)
```

### Example 2: Database Integration

```python
import sqlite3
from datetime import datetime
import pandas as pd

class RiskDatabase:
    """Store and retrieve historical risk data."""
    
    def __init__(self, db_path="risk_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS risk_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                coin_id TEXT,
                symbol TEXT,
                name TEXT,
                risk_score REAL,
                volatility_proxy REAL,
                price_change_24h REAL,
                risk_reason TEXT,
                current_price REAL,
                market_cap REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_risk_data(self, df_risk):
        """Store current risk data."""
        conn = sqlite3.connect(self.db_path)
        
        # Prepare data
        timestamp = datetime.now().isoformat()
        
        for _, row in df_risk.iterrows():
            conn.execute('''
                INSERT INTO risk_history 
                (timestamp, coin_id, symbol, name, risk_score, volatility_proxy, 
                 price_change_24h, risk_reason, current_price, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                row.get('id', ''),
                row.get('symbol', ''),
                row.get('name', ''),
                row.get('risk_score', 0),
                row.get('volatility_proxy', 0),
                row.get('price_change_percentage_24h', 0),
                row.get('risk_reason', ''),
                row.get('current_price', 0),
                row.get('market_cap', 0)
            ))
        
        conn.commit()
        conn.close()
        print(f"Stored risk data for {len(df_risk)} coins")
    
    def get_risk_history(self, symbol, days=7):
        """Get historical risk data for a specific coin."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM risk_history 
            WHERE symbol = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        return df
    
    def get_risk_trends(self, days=30):
        """Get risk trends across all coins."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                DATE(timestamp) as date,
                AVG(risk_score) as avg_risk,
                MAX(risk_score) as max_risk,
                COUNT(CASE WHEN risk_score >= 70 THEN 1 END) as high_risk_count
            FROM risk_history 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

# Usage example
risk_db = RiskDatabase()

# Store current risk data
df, _, _ = get_markets_with_fallback(max_coins=100)
df_risk = build_features_and_risk(df)
risk_db.store_risk_data(df_risk)

# Retrieve historical data
btc_history = risk_db.get_risk_history('BTC', days=30)
print(f"BTC risk history: {len(btc_history)} records")

# Get market trends
trends = risk_db.get_risk_trends(days=7)
print("Recent risk trends:")
print(trends)
```

### Example 3: API Endpoint Creation

```python
from flask import Flask, jsonify, request
from app import get_markets_with_fallback, build_features_and_risk
import pandas as pd

app = Flask(__name__)

@app.route('/api/risk-scores', methods=['GET'])
def get_risk_scores():
    """API endpoint to get current risk scores."""
    
    try:
        # Get parameters
        max_coins = request.args.get('max_coins', 50, type=int)
        threshold = request.args.get('threshold', 0, type=float)
        source = request.args.get('source', 'auto')
        
        # Fetch and process data
        df, source_used, error = get_markets_with_fallback(
            source=source, 
            max_coins=max_coins
        )
        
        df_risk = build_features_and_risk(df)
        
        # Filter by threshold if specified
        if threshold > 0:
            df_risk = df_risk[df_risk['risk_score'] >= threshold]
        
        # Convert to JSON-friendly format
        result = {
            'status': 'success',
            'data_source': source_used,
            'total_coins': len(df_risk),
            'coins': df_risk[[
                'name', 'symbol', 'current_price', 'risk_score', 
                'volatility_proxy', 'price_change_percentage_24h', 'risk_reason'
            ]].to_dict('records'),
            'error': str(error) if error else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/coin/<symbol>/risk', methods=['GET'])
def get_coin_risk(symbol):
    """Get risk data for a specific coin."""
    
    try:
        df, source_used, error = get_markets_with_fallback(max_coins=200)
        df_risk = build_features_and_risk(df)
        
        coin_data = df_risk[df_risk['symbol'].str.upper() == symbol.upper()]
        
        if coin_data.empty:
            return jsonify({
                'status': 'error',
                'message': f'Coin {symbol} not found'
            }), 404
        
        coin = coin_data.iloc[0]
        
        result = {
            'status': 'success',
            'data_source': source_used,
            'coin': {
                'name': coin['name'],
                'symbol': coin['symbol'],
                'current_price': coin['current_price'],
                'risk_score': coin['risk_score'],
                'risk_reason': coin['risk_reason'],
                'volatility_proxy': coin['volatility_proxy'],
                'price_changes': {
                    '1h': coin.get('price_change_percentage_1h'),
                    '24h': coin.get('price_change_percentage_24h'),
                    '7d': coin.get('price_change_percentage_7d'),
                    '30d': coin.get('price_change_percentage_30d')
                },
                'market_data': {
                    'market_cap': coin['market_cap'],
                    'volume_24h': coin['volume_24h']
                }
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

These examples demonstrate comprehensive usage patterns for the Crashcaster application, from basic data fetching to advanced integrations with external systems. Each example includes error handling, configuration options, and practical use cases that can be adapted for specific requirements.