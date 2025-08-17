# CrashCaster — Early Warning for Crypto Crashes

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue)](https://crashcaster.streamlit.app/)

Welcome to **CrashCaster**, a real-time cryptocurrency crash risk monitoring tool built with Streamlit. This app analyzes market data from CoinGecko to identify at-risk coins, visualize risk scores, and provide actionable recommendations. Perfect for crypto enthusiasts and traders looking to stay ahead of market volatility!

## Features
- **Risk Assessment**: Calculates a risk score (0–100) based on volatility, price changes, and volume-to-market-cap ratio.
- **Real-Time Data**: Fetches live data from CoinGecko with caching and offline fallback options.
- **Interactive UI**: Explore top-risk coins, analyze individual coins with sparkline charts, and view risk distributions.
- **Recommendations**: Offers "trend" and "reversal" strategies to suggest low-risk investment opportunities.
- **Customizable Settings**: Adjust the number of coins, risk threshold, and data source via the sidebar.

## Demo
Check out the live app: [CrashCaster on Streamlit Cloud](https://crashcaster.streamlit.app/)

## Installation
To run CrashCaster locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crashcaster.git
   cd crashcaster
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- **Sidebar Settings**: Select "offline", "auto", or "primary" data source. Adjust the number of coins (10–50) and risk threshold (0–100).
- **Dashboard**: View at-risk coins, all coins, and interactive charts (Top-10 Risk, Risk Distribution, 24h Movers).
- **Coin Analysis**: Pick a coin and click "Analyze" to see its risk score, sparkline, and detailed metrics.
- **Recommendations**: Choose "trend" or "reversal" strategy to get suggested coins.

## Requirements
- Python 3.11+
- Dependencies (listed in `requirements.txt`):
  - `streamlit==1.39.0`
  - `pandas==2.2.2`
  - `numpy==2.0.2`
  - `plotly==5.24.1`
  - `requests==2.32.3`

## Offline Mode
- The app includes an offline snapshot feature. Use the "Refresh offline snapshot" button in the sidebar to update `offline/markets_sample.json`.
- Commit this file to the repo after refreshing for offline testing. Switch to "offline" mode in the sidebar for demo reliability.

## Contributing
Feel free to fork the repository, submit issues, or suggest enhancements. Pull requests are welcome!

## License
[MIT License](LICENSE) - Free to use and modify.

## Acknowledgements
- Data provided by [CoinGecko API](https://www.coingecko.com/en/api).
- Built with [Streamlit](https://streamlit.io/) for an interactive web interface.
- Developed by Athika, Shreya, Dhruv.

## Contact
For questions or feedback, reach out via the GitHub Issues tab.
