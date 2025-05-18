##![Forex Dashboard arcitecture]
![deepseek_mermaid_20250518_62caac](https://github.com/user-attachments/assets/437c9cf5-836b-48c4-9a8a-97d689695185)# AI-Powered Forex Trading Dashboard

![Forex Dashboard arcitecture](static/images/dashboard-screenshot.png) 

## 🚀 Overview
An intelligent Forex trading platform that combines real-time market data analysis with AI-driven pattern recognition and risk management recommendations. The system analyzes currency pairs, identifies trading opportunities using technical indicators and Gemini AI, and provides actionable insights with proper risk parameters.

## 🔥 Key Features
- **AI-Powered Analysis**: Gemini AI identifies chart patterns and trading opportunities
- **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic Oscillator
- **Risk Management**: AI-generated position sizing and risk-reward strategies
- **Interactive Visualization**: Dynamic charts with entry/exit points visualization
- **Real-time Data**: Polygon.io API integration for market data

## 🛠 Tech Stack
| Category        | Technologies |
|-----------------|--------------|
| **Backend**     | Python, Flask |
| **AI/ML Core**  | Gemini API, Technical Analysis Algorithms |
| **Frontend**    | HTML5, CSS3, JavaScript, Chart.js |
| **Data**        | Polygon.io API, Pandas, NumPy |
| **DevOps**      | Git, Pipenv |

# 🧠 AI Trading Assistant

An intelligent trading assistant that leverages AI for technical analysis, pattern recognition, strategy generation, and risk management. The system is designed to analyze Forex markets using OHLC data and make informed trading decisions.

---

## 🚀 AI Workflow Details

### 📈 1. Data Acquisition
- Fetches Open-High-Low-Close (OHLC) data from the **Polygon.io** API.

---

### ⚙️ 2. Feature Engineering
- Calculates **20+ technical indicators** (e.g., RSI, MACD, Bollinger Bands).
- Normalizes features for improved AI model performance.

---

### 🔍 3. Pattern Recognition
- **Gemini AI** analyzes price action and indicator convergence.
- Detects classic chart patterns:
  - Head & Shoulders (H&S)
  - Double Tops / Bottoms
  - Flags and Pennants

---

### 🧠 4. Strategy Formulation
- AI recommends:
  - **Optimal entry/exit points**
  - **Position sizing strategies** based on risk and confidence levels

---

### 🛡️ 5. Risk Management
- AI implements portfolio-level risk constraints.
- Enforces a **2% maximum risk per trade** rule.

---

## 📂 Project Structure 


## 📁 Root Directory

```
forex_trading_dashboard/
├── app.py
├── config.py
├── requirements.txt
├── forex_trader.py
├── static/
├── templates/
```

---

### 📄 app.py

- **Purpose:** Entry point of the Flask web application.
- **Responsibility:** Handles routing, rendering templates, and integrating trading logic.

---

### ⚙️ config.py

- **Purpose:** Stores configuration variables such as API keys and constants.
- **Tip:** Keep sensitive info out of version control (use `.env` for secrets).

---

### 📦 requirements.txt

- **Purpose:** Lists all Python dependencies.
- **Use:** Run `pip install -r requirements.txt` to install required packages.

---

### 📊 forex_trader.py

- **Purpose:** Core logic for:
  - Fetching data
  - Calculating technical indicators
  - Performing pattern recognition
  - Generating trade signals and risk constraints

---

## 🎨 static/

Contains static assets like CSS, JavaScript, and images.

```
static/
├── css/
│   └── styles.css          # Custom styles for UI
├── js/
│   └── scripts.js          # Custom interactivity/animations
└── images/                 # Generated or static charts/images
```

---

## 🧩 templates/

Contains HTML templates rendered by Flask.

```
templates/
├── base.html               # Base layout (header/footer)
├── index.html              # Homepage/dashboard view
├── analysis.html           # Detailed market analysis page
└── risk.html               # Risk management visualization
```

---

### ✅ Summary

This structure promotes **modularity**, **clarity**, and **scalability**:

- 👨‍💻 **app.py** runs the server
- 🧠 **forex_trader.py** handles trading intelligence
- 🎨 **static/** and **templates/** define your UI
- ⚙️ **config.py** and **requirements.txt** handle setup and environment

---

Feel free to customize or expand this structure based on project needs (e.g., adding `tests/`, `logs/`, `notebooks/`).ng analysis code (modified)
