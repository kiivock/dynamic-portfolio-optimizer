# Dynamic Portfolio Optimizer

A smart ETF portfolio optimization tool using **Machine Learning** (LSTM) for return forecasting and **Modern Portfolio Theory** for optimal allocation.

## Features

- **ML-Powered Forecasting**: Individual LSTM models trained per ETF ticker (2000 epochs)
- **Portfolio Optimization**: Multiple strategies (Max Sharpe, Min Volatility, Risk Parity)
- **Interactive Dashboard**: Streamlit-based UI with charts and recommendations
- **REST API**: FastAPI backend for programmatic access

## Supported ETFs

| Ticker | Sector | Region |
|--------|--------|--------|
| PSI | Semiconductors | North America |
| IYW | US Technology | North America |
| RING | Gold Miners | Developed Markets |
| PICK | Metals & Mining | Developed Markets |
| NLR | Nuclear Energy | Developed Markets |
| UTES | Utilities | North America |
| LIT | Lithium & Battery | Developed Markets |
| NANR | Natural Resources | North America |
| GUNR | Global Resources | Developed Markets |
| XCEM | Emerging Markets | Emerging Markets |
| PTLC | Large Cap | North America |
| FXU | Utilities Alpha | North America |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the API Backend

```bash
uvicorn backend.api:app --reload
```

API available at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 2. Start the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

Dashboard available at: http://localhost:8501

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/forecast` | POST | Get 22-day return forecasts |
| `/smart-invest` | POST | Combined forecast + optimization |
| `/optimize` | POST | Basic portfolio optimization |
| `/efficient-frontier` | POST | Compute efficient frontier |
| `/chart-data` | POST | Historical price data |

## Model Architecture

- **Type**: LSTM Neural Network
- **Input**: 10-day sequences with 9 features
- **Features**: Momentum (1m, 3m, 6m), Volatility, RSI, Position in 52-week range, Correlation, Max Drawdown, Region
- **Output**: Predicted 22-day return
- **Training**: 2000 epochs per ticker

## Project Structure

```
dynamic_portfolio_optimizer/
├── backend/
│   ├── __init__.py
│   ├── api.py          # FastAPI endpoints
│   ├── forecaster.py   # ML prediction logic
│   └── optimizer.py    # Portfolio optimization
├── models/
│   ├── *_model.keras   # 12 LSTM models (one per ETF)
│   └── scalers.pkl     # Per-ticker scalers
├── streamlit_app.py    # Frontend dashboard
├── requirements.txt
└── README.md
```

## Tech Stack

- **Backend**: FastAPI, TensorFlow/Keras, NumPy, Pandas
- **Frontend**: Streamlit, Altair
- **Data**: Yahoo Finance API
- **ML**: LSTM with per-ticker training

## License

MIT
