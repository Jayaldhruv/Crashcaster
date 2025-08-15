# Crashcaster Developer Guide

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [Code Organization](#code-organization)
4. [Development Workflow](#development-workflow)
5. [Testing Guidelines](#testing-guidelines)
6. [Performance Optimization](#performance-optimization)
7. [Deployment](#deployment)
8. [Contributing Guidelines](#contributing-guidelines)

---

## Development Environment Setup

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** for version control
- **Virtual environment** (venv, conda, or virtualenv)
- **Code editor** (VS Code, PyCharm, or similar)

### Local Development Setup

#### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/your-username/crashcaster.git
cd crashcaster

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Development Dependencies

Create a `requirements-dev.txt` file for development tools:

```txt
# Development dependencies
black==23.3.0          # Code formatting
flake8==6.0.0          # Linting
pytest==7.3.1          # Testing framework
pytest-cov==4.1.0      # Coverage reporting
pre-commit==3.3.2      # Git hooks
mypy==1.3.0            # Type checking
streamlit-profiler==0.2.0  # Performance profiling
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

#### 3. IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        "cache/*.json": true
    }
}
```

#### 4. Pre-commit Hooks

Set up pre-commit hooks for code quality:

```bash
# Install pre-commit
pre-commit install

# Create .pre-commit-config.yaml
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-python-dateutil]
```

### Environment Variables

Create a `.env` file for local development:

```bash
# Optional CoinGecko API configuration
COINGECKO_API_KEY=your_api_key_here
COINGECKO_BASE_URL=https://api.coingecko.com/api/v3

# Cache configuration
CACHE_TTL=600
REQUEST_DELAY=2.0
TIMEOUT=8

# Development settings
DEBUG=true
LOG_LEVEL=INFO

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Crashcaster Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   Data Layer    │    │   Business Logic │    │ Presentation│ │
│  │                 │    │                  │    │    Layer    │ │
│  │ • CoinGecko API │───▶│ • Risk Engine    │───▶│ • Streamlit │ │
│  │ • Cache System  │    │ • Feature Eng.   │    │ • Dashboard │ │
│  │ • Offline Store │    │ • Normalization  │    │ • Components│ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   Utilities     │    │   Configuration  │    │   Styling   │ │
│  │                 │    │                  │    │             │ │
│  │ • Caching       │    │ • Constants      │    │ • CSS       │ │
│  │ • Error Handle  │    │ • Settings       │    │ • Themes    │ │
│  │ • Validation    │    │ • Logging        │    │ • Badges    │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### Data Layer Components

1. **API Integration** (`fetch_cg_markets`, `norm_from_cg`)
   - CoinGecko API client
   - Rate limiting and error handling
   - Data normalization and validation

2. **Caching System** (`_read_cache`, `_write_cache`)
   - Disk-based JSON cache
   - TTL-based expiration
   - Cache key management

3. **Offline Storage** (`load_offline_json`)
   - Fallback data source
   - JSON-based storage
   - Automatic snapshot creation

#### Business Logic Layer

1. **Risk Engine** (`build_features_and_risk`)
   - Feature engineering pipeline
   - Risk score calculation
   - Reason generation

2. **Statistical Processing** (`_norm`, `_coerce_numeric`)
   - Z-score normalization
   - Sigmoid transformation
   - Data type coercion

3. **Market Analysis** (`pct_change_from_sparkline`)
   - Price change calculations
   - Trend analysis
   - Multi-timeframe processing

#### Presentation Layer

1. **Streamlit UI** (main application)
   - Dashboard layout
   - Interactive components
   - Real-time updates

2. **Visualization** (Plotly integration)
   - Risk gauges
   - Sparkline charts
   - Interactive plots

3. **Styling** (`risk_badge_class`, `reason_badges`)
   - CSS styling
   - Theme management
   - Responsive design

### Data Flow

```
1. User Request
    ↓
2. Data Source Selection (auto/primary/offline)
    ↓
3. Cache Check
    ↓
4. API Fetch (if needed)
    ↓
5. Data Normalization
    ↓
6. Feature Engineering
    ↓
7. Risk Score Calculation
    ↓
8. UI Rendering
    ↓
9. User Interaction
```

---

## Code Organization

### File Structure

```
crashcaster/
├── app.py                      # Main application entry point
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── runtime.txt                # Python version for deployment
├── .env                       # Environment variables (local)
├── .gitignore                 # Git ignore rules
├── .pre-commit-config.yaml    # Pre-commit hooks
├── README.md                  # Project documentation
├── LICENSE                    # MIT license
│
├── .streamlit/
│   └── config.toml           # Streamlit configuration
│
├── docs/
│   ├── API_DOCUMENTATION.md   # API reference
│   ├── USAGE_EXAMPLES.md     # Usage examples
│   └── DEVELOPER_GUIDE.md    # This file
│
├── cache/                    # API response cache
│   └── .gitkeep
│
├── offline/                  # Offline data storage
│   └── .gitkeep
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_data_fetching.py
│   ├── test_risk_engine.py
│   └── test_ui_components.py
│
├── src/                      # Modular source code (future)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── api_client.py
│   │   └── cache_manager.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── models.py
│   └── ui/
│       ├── __init__.py
│       ├── components.py
│       └── styling.py
│
└── scripts/                  # Utility scripts
    ├── setup_dev.py         # Development environment setup
    ├── data_migration.py    # Data migration utilities
    └── performance_test.py  # Performance testing
```

### Code Style Guidelines

#### Python Style

Follow **PEP 8** with these specific guidelines:

```python
# Line length: 88 characters (Black default)
# Use type hints where appropriate
def fetch_cg_markets(
    max_coins: int = 30, 
    currency: str = "usd", 
    ttl: int = CACHE_TTL
) -> Tuple[List[Dict], str]:
    """Fetch cryptocurrency market data with caching."""
    pass

# Use descriptive variable names
risk_score_threshold = 70  # Good
threshold = 70             # Less clear

# Function naming: snake_case
def calculate_risk_score():
    pass

# Class naming: PascalCase
class RiskEngine:
    pass

# Constants: UPPER_SNAKE_CASE
BASE_URL_CG = "https://api.coingecko.com/api/v3"
```

#### Documentation Standards

```python
def build_features_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk scores and features for cryptocurrency data.
    
    This function performs feature engineering and risk scoring on market data,
    including volatility analysis, momentum calculations, and risk categorization.
    
    Args:
        df (pd.DataFrame): Market data with price and volume information.
            Required columns: current_price, market_cap, volume_24h,
            price_change_percentage_1h, price_change_percentage_24h,
            price_change_percentage_7d
    
    Returns:
        pd.DataFrame: Enhanced dataframe with additional columns:
            - risk_score (float): Risk score from 0-100
            - risk_reason (str): Human-readable risk explanation
            - volatility_proxy (float): Composite volatility measure
            - abs_24h (float): Absolute 24-hour price change
            - down_1h (float): Downward 1-hour movement
            - volume_ratio (float): Volume to market cap ratio
    
    Raises:
        ValueError: If required columns are missing from input DataFrame
        
    Example:
        >>> df = get_markets_with_fallback(max_coins=10)
        >>> df_risk = build_features_and_risk(df)
        >>> high_risk = df_risk[df_risk['risk_score'] >= 70]
    """
    pass
```

### Error Handling Patterns

```python
# Use specific exception types
try:
    data = fetch_api_data()
except requests.RequestException as e:
    logger.error(f"API request failed: {e}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON response: {e}")
    return None

# Graceful degradation
def get_data_with_fallback():
    try:
        return fetch_live_data()
    except Exception as e:
        logger.warning(f"Live data failed, using cache: {e}")
        return load_cached_data()

# Input validation
def validate_risk_threshold(threshold: float) -> float:
    if not 0 <= threshold <= 100:
        raise ValueError("Risk threshold must be between 0 and 100")
    return threshold
```

---

## Development Workflow

### Git Workflow

#### Branch Strategy

```
main                    # Production-ready code
├── develop            # Integration branch
├── feature/risk-v2    # Feature branches
├── hotfix/cache-bug   # Hotfix branches
└── release/v1.2.0     # Release branches
```

#### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(risk): add custom risk model support

- Implement CustomRiskModel class
- Add configurable weight system
- Update risk calculation pipeline

Closes #123
```

### Development Process

#### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-risk-model

# Make changes and commit
git add .
git commit -m "feat(risk): implement custom risk model"

# Push and create PR
git push origin feature/new-risk-model
```

#### 2. Code Review Checklist

**Functionality:**
- [ ] Feature works as intended
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Performance impact is acceptable

**Code Quality:**
- [ ] Code follows style guidelines
- [ ] Functions are properly documented
- [ ] Variable names are descriptive
- [ ] No code duplication

**Testing:**
- [ ] Unit tests are included
- [ ] Tests cover edge cases
- [ ] All tests pass
- [ ] Coverage is maintained

**Documentation:**
- [ ] API documentation is updated
- [ ] Usage examples are provided
- [ ] README is updated if needed

#### 3. Release Process

```bash
# Create release branch
git checkout -b release/v1.2.0

# Update version numbers
# Update CHANGELOG.md
# Final testing

# Merge to main and develop
git checkout main
git merge release/v1.2.0
git tag v1.2.0

git checkout develop
git merge release/v1.2.0
```

### Local Development Commands

#### Development Server

```bash
# Run with hot reload
streamlit run app.py

# Run with specific port
streamlit run app.py --server.port 8502

# Run with debug mode
streamlit run app.py --logger.level debug
```

#### Code Quality

```bash
# Format code
black app.py

# Lint code
flake8 app.py

# Type checking
mypy app.py

# Run all quality checks
pre-commit run --all-files
```

#### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_risk_engine.py

# Run with verbose output
pytest -v
```

---

## Testing Guidelines

### Test Structure

#### Unit Tests Example

```python
# tests/test_risk_engine.py
import pytest
import pandas as pd
import numpy as np
from app import build_features_and_risk, _norm, pct_change_from_sparkline


class TestRiskEngine:
    """Test suite for risk scoring engine."""
    
    def test_build_features_and_risk_basic(self):
        """Test basic risk calculation functionality."""
        # Arrange
        df = pd.DataFrame({
            'current_price': [50000, 3000],
            'market_cap': [1000000000, 500000000],
            'volume_24h': [50000000, 25000000],
            'price_change_percentage_1h': [1.5, -2.0],
            'price_change_percentage_24h': [5.0, -8.0],
            'price_change_percentage_7d': [10.0, -15.0]
        })
        
        # Act
        result = build_features_and_risk(df)
        
        # Assert
        assert 'risk_score' in result.columns
        assert 'risk_reason' in result.columns
        assert all(0 <= score <= 100 for score in result['risk_score'])
        assert len(result) == len(df)
    
    def test_norm_function(self):
        """Test statistical normalization function."""
        # Arrange
        s = pd.Series([1, 2, 3, 4, 5])
        
        # Act
        result = _norm(s)
        
        # Assert
        assert all(0 <= val <= 1 for val in result)
        assert len(result) == len(s)
    
    def test_pct_change_from_sparkline(self):
        """Test percentage change calculation from price array."""
        # Arrange
        prices = [100, 102, 98, 105, 103]
        
        # Act
        result = pct_change_from_sparkline(prices, 2)
        
        # Assert
        assert isinstance(result, (float, type(None)))
        if result is not None:
            assert -100 <= result <= 1000  # Reasonable bounds
    
    def test_risk_score_edge_cases(self):
        """Test risk scoring with edge cases."""
        # Test with missing data
        df_missing = pd.DataFrame({
            'current_price': [50000],
            'market_cap': [np.nan],
            'volume_24h': [None],
            'price_change_percentage_1h': [np.nan],
            'price_change_percentage_24h': [None],
            'price_change_percentage_7d': [np.nan]
        })
        
        result = build_features_and_risk(df_missing)
        assert 'risk_score' in result.columns
        assert not result['risk_score'].isna().all()


class TestDataFetching:
    """Test suite for data fetching functionality."""
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock CoinGecko API response."""
        return [
            {
                'id': 'bitcoin',
                'symbol': 'btc',
                'name': 'Bitcoin',
                'current_price': 50000,
                'market_cap': 1000000000,
                'total_volume': 50000000,
                'price_change_percentage_24h': 5.0,
                'sparkline': {'price': [49000, 50000, 51000]}
            }
        ]
    
    def test_norm_from_cg(self, mock_api_response):
        """Test CoinGecko data normalization."""
        from app import norm_from_cg
        
        result = norm_from_cg(mock_api_response[0])
        
        assert result['symbol'] == 'BTC'  # Should be uppercase
        assert result['current_price'] == 50000
        assert 'sparkline' in result
```

#### Integration Tests

```python
# tests/test_integration.py
import pytest
from app import get_markets_with_fallback, build_features_and_risk


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_risk_analysis_pipeline(self):
        """Test complete risk analysis from data fetch to scoring."""
        # This test uses offline mode to avoid API dependencies
        df, source, error = get_markets_with_fallback(
            source="offline", 
            max_coins=10
        )
        
        if not df.empty:
            df_risk = build_features_and_risk(df)
            
            # Verify pipeline output
            assert 'risk_score' in df_risk.columns
            assert 'risk_reason' in df_risk.columns
            assert len(df_risk) > 0
            assert source == "OFFLINE(JSON)"
```

#### Performance Tests

```python
# tests/test_performance.py
import time
import pytest
from app import build_features_and_risk, get_markets_with_fallback


class TestPerformance:
    """Performance tests for critical functions."""
    
    def test_risk_calculation_performance(self):
        """Test risk calculation performance with large datasets."""
        df, _, _ = get_markets_with_fallback(max_coins=100)
        
        start_time = time.time()
        result = build_features_and_risk(df)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(result) == len(df)
```

### Test Configuration

#### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=app
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

---

## Performance Optimization

### Profiling

#### Streamlit Profiler

```python
# Add to development version
import streamlit as st
from streamlit_profiler import Profiler

with Profiler():
    # Your Streamlit code here
    df = get_markets_with_fallback(max_coins=100)
    df_risk = build_features_and_risk(df)
    st.dataframe(df_risk)
```

#### Memory Profiling

```python
# scripts/memory_profile.py
from memory_profiler import profile
from app import get_markets_with_fallback, build_features_and_risk

@profile
def profile_risk_calculation():
    df, _, _ = get_markets_with_fallback(max_coins=100)
    df_risk = build_features_and_risk(df)
    return df_risk

if __name__ == "__main__":
    profile_risk_calculation()
```

### Optimization Strategies

#### 1. Caching Optimization

```python
# Implement smarter caching
class SmartCache:
    def __init__(self, max_size=100, ttl=600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
```

#### 2. Data Processing Optimization

```python
# Vectorized operations
def optimized_risk_calculation(df):
    """Optimized version using vectorized pandas operations."""
    # Use .loc for better performance
    df.loc[:, 'abs_24h'] = df['price_change_percentage_24h'].abs()
    
    # Vectorized calculations
    df.loc[:, 'volatility_proxy'] = (
        df['price_change_percentage_1h'].abs().fillna(0) +
        df['price_change_percentage_24h'].abs().fillna(0) +
        df['price_change_percentage_7d'].abs().fillna(0)
    ) / 3.0
    
    return df
```

#### 3. API Request Optimization

```python
# Batch API requests
async def fetch_multiple_currencies(currencies, max_coins=30):
    """Fetch data for multiple currencies concurrently."""
    import asyncio
    import aiohttp
    
    async def fetch_currency(session, currency):
        url = f"{BASE_URL_CG}/coins/markets"
        params = {
            'vs_currency': currency,
            'order': 'market_cap_desc',
            'per_page': max_coins,
            'page': 1,
            'sparkline': True
        }
        async with session.get(url, params=params) as response:
            return await response.json()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_currency(session, currency) for currency in currencies]
        results = await asyncio.gather(*tasks)
        return dict(zip(currencies, results))
```

---

## Deployment

### Production Deployment

#### Streamlit Cloud

```toml
# .streamlit/config.toml (production)
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#FF4B4B"
```

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directories
RUN mkdir -p cache offline

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  crashcaster:
    build: .
    ports:
      - "8501:8501"
    environment:
      - CACHE_TTL=600
      - REQUEST_DELAY=2.0
    volumes:
      - ./cache:/app/cache
      - ./offline:/app/offline
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - crashcaster
    restart: unless-stopped
```

### Monitoring and Logging

#### Application Logging

```python
# Add to app.py
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/crashcaster.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use throughout application
def fetch_cg_markets(max_coins=30, currency="usd", ttl=CACHE_TTL):
    logger.info(f"Fetching {max_coins} coins in {currency}")
    try:
        # ... existing code ...
        logger.info(f"Successfully fetched {len(data)} coins from API")
        return data, "LIVE(CG)"
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        raise
```

#### Health Monitoring

```python
# health_check.py
import requests
import json
from datetime import datetime

def health_check():
    """Comprehensive health check for the application."""
    checks = {
        'timestamp': datetime.now().isoformat(),
        'status': 'healthy',
        'checks': {}
    }
    
    # Check Streamlit app
    try:
        response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
        checks['checks']['streamlit'] = {
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'response_time': response.elapsed.total_seconds()
        }
    except Exception as e:
        checks['checks']['streamlit'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check CoinGecko API
    try:
        response = requests.get('https://api.coingecko.com/api/v3/ping', timeout=5)
        checks['checks']['coingecko_api'] = {
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'response_time': response.elapsed.total_seconds()
        }
    except Exception as e:
        checks['checks']['coingecko_api'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check cache directory
    try:
        import os
        cache_size = sum(os.path.getsize(os.path.join('cache', f)) 
                        for f in os.listdir('cache') if f.endswith('.json'))
        checks['checks']['cache'] = {
            'status': 'healthy',
            'size_mb': cache_size / 1024 / 1024
        }
    except Exception as e:
        checks['checks']['cache'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Overall status
    if any(check.get('status') == 'unhealthy' for check in checks['checks'].values()):
        checks['status'] = 'unhealthy'
    
    return checks

if __name__ == "__main__":
    print(json.dumps(health_check(), indent=2))
```

---

## Contributing Guidelines

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `develop`
4. **Make your changes** with tests
5. **Submit a pull request**

### Contribution Types

#### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Windows 10]
- Python Version: [e.g. 3.9]
- Streamlit Version: [e.g. 1.37.0]
```

#### Feature Requests

```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other solutions you've considered.
```

#### Code Contributions

**Pull Request Template:**

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

### Code Review Process

1. **Automated Checks** must pass (linting, tests)
2. **Manual Review** by maintainer
3. **Testing** in development environment
4. **Approval** and merge

### Recognition

Contributors are recognized in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- GitHub releases for major features

This developer guide provides a comprehensive foundation for contributing to and maintaining the Crashcaster project. It emphasizes code quality, testing, and collaborative development practices.