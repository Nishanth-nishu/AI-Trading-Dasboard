import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import google.generativeai as genai
import time
import json
import os
import config

# Configure Gemini AI
genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

class ForexTrader:
    def __init__(self):
        self.data = {}
        self.analysis = {}
        self.recommendations = {}
        self.risk_management = {}

    def fetch_forex_data(self, symbol, timespan="day", multiplier=1, from_date=None, to_date=None):
        """Fetch historical forex data from Polygon.io"""
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')

        ticker = f"C:{symbol}"
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=120&apiKey={config.POLYGON_API_KEY}"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                # Convert to DataFrame
                df = pd.DataFrame(data['results'])

                # Rename columns to more readable format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    't': 'timestamp'
                })

                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # Calculate additional technical indicators
                self.add_technical_indicators(df)

                self.data[symbol] = df
                print(f"Successfully fetched data for {symbol}")
                return df
            else:
                print(f"No results found for {symbol}")
                return None
        else:
            print(f"Error fetching data for {symbol}: {response.status_code}")
            print(response.text)
            return None

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()

        return df
    
    def clean_gemini_response(self,text):
        """Remove markdown formatting from Gemini responses"""
        # Remove ** markers
        text = text.replace('**', '')
        text = text.replace('*', '')
        return text.strip()
    
    def identify_patterns(self, symbol):
        """Use Gemini AI to identify patterns in the forex data"""
        if symbol not in self.data or self.data[symbol] is None:
            print(f"No data available for {symbol}")
            return None

        df = self.data[symbol].copy()

        # Prepare the last 10 days of data for analysis
        recent_data = df.tail(10).copy()
        recent_data_json = recent_data[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                      'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
                                      '%K', '%D', 'ATR']].to_json(orient='records', date_format='iso')

        # Calculate key statistics
        current_price = df['close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        stoch_k = df['%K'].iloc[-1]
        stoch_d = df['%D'].iloc[-1]

        # Prepare a prompt for Gemini AI
        prompt = f"""
        Analyze the following forex data for {symbol} and identify trading patterns, potential entry/exit points,
        and risk management strategies. Provide a detailed analysis and trading recommendation.

        Current Technical Indicators:
        - Current Price: {current_price:.4f}
        - SMA 20: {sma_20:.4f}
        - SMA 50: {sma_50:.4f}
        - RSI (14): {rsi:.2f}
        - MACD: {macd:.4f}
        - MACD Signal: {macd_signal:.4f}
        - Stochastic %K: {stoch_k:.2f}
        - Stochastic %D: {stoch_d:.2f}

        Based on this data:
        1. Identify any recognizable chart patterns (e.g., head and shoulders, double top/bottom, flags, etc.)
        2. Suggest an appropriate trading strategy (trend following, mean reversion, breakout)
        3. Provide entry point recommendations with specific price levels
        4. Recommend stop loss placement (in pips or price)
        5. Provide take profit targets (in pips or price)
        6. Estimate risk-reward ratio
        7. Suggest position sizing based on 2% max risk per trade

        Your analysis should be clear, quantitative, and actionable, focused on the next 1-5 trading days.
        """

        try:
            # Query Gemini AI
            response = model.generate_content(prompt)
            analysis_text = response.text
            analysis_text = self.clean_gemini_response(analysis_text)


            # Store the analysis
            self.analysis[symbol] = analysis_text

            # Extract recommendations using another Gemini query
            recommendation_prompt = f"""
            Based on the following forex analysis for {symbol}, extract the key trading recommendations
            in a structured format. Return your response as a JSON object with the following fields:

            Analysis: {analysis_text}

            Format:
            {{
                "recommendation": "BUY" or "SELL" or "NEUTRAL",
                "strategy": "brief description of strategy",
                "entry_price": numerical value,
                "stop_loss": numerical value,
                "take_profit": numerical value,
                "risk_reward_ratio": numerical value,
                "position_sizing_percentage": numerical value,
                "timeframe": "short-term/medium-term/long-term",
                "confidence": "low/medium/high"
            }}

            Provide only the JSON object with no additional text.
            """

            recommendation_response = model.generate_content(recommendation_prompt)
            recommendation_text = recommendation_response.text

            # Clean up the response text to extract just the JSON part
            recommendation_text = recommendation_text.strip()
            if "```json" in recommendation_text:
                recommendation_text = recommendation_text.split("```json")[1].split("```")[0].strip()
            elif "```" in recommendation_text:
                recommendation_text = recommendation_text.split("```")[1].strip()

            # Parse JSON
            recommendation_json = json.loads(recommendation_text)

            # Validate recommendation data
            required_fields = ['recommendation', 'strategy', 'entry_price',
                             'stop_loss', 'take_profit', 'risk_reward_ratio',
                             'position_sizing_percentage', 'timeframe', 'confidence']

            for field in required_fields:
                if field not in recommendation_json:
                    print(f"Warning: Missing field {field} in recommendation for {symbol}")
                    recommendation_json[field] = None
                elif recommendation_json[field] is None:
                    print(f"Warning: Null value for field {field} in recommendation for {symbol}")

            self.recommendations[symbol] = recommendation_json

            return analysis_text

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON recommendation for {symbol}: {e}")
            print("Original response:")
            print(recommendation_text)
            return f"Error: Failed to parse recommendation - {e}"
        except Exception as e:
            print(f"Error analyzing {symbol} with Gemini AI: {e}")
            return f"Error: {e}"

    def generate_risk_management_plan(self):
        """Generate overall risk management plan using Gemini AI"""
        if not self.recommendations:
            print("No recommendations available for risk management plan")
            return None

        # Prepare a summary of recommendations
        recommendations_summary = ""
        for symbol, rec in self.recommendations.items():
            recommendations_summary += f"{symbol}: {rec.get('recommendation', 'N/A')} - Entry: {rec.get('entry_price', 'N/A')}, SL: {rec.get('stop_loss', 'N/A')}, TP: {rec.get('take_profit', 'N/A')}\n"

        # Prepare a prompt for Gemini AI
        prompt = f"""
        Based on the following forex trading recommendations, create a comprehensive risk management plan.
        The plan should detail how to manage multiple positions across different currency pairs,
        addressing portfolio exposure, correlation risk, and overall risk limits.

        Current Recommendations:
        {recommendations_summary}

        Please provide:
        1. Maximum portfolio risk percentage (total capital at risk)
        2. Maximum correlated exposure (max percentage in correlated pairs)
        3. Position sizing rules with examples
        4. Criteria for adjusting stop losses
        5. Rules for partial profit-taking
        6. Criteria for exiting all positions (market conditions)
        7. Daily/weekly loss limits that would trigger trading pause
        8. Rules for scaling in/out of positions

        Important Points :
        **PNP STRATEGY Forex Pairs Use 200EMA on a 5min Timeframe with default indicator setting
        **Use 200EMA on a 5min Timeframe with 15Min


        1. Look for specific patterns
        a. Double Tops/Bottoms (With proper V shape in the
        center)
        b. Head & Shoulders/Inverted HNS
        2. Entry timing: 10Am to 10 Pm.
        3. Use both 5 Min & 15 Min 200EMA on the 5-minute chart.
        4. For Head & Shoulders only enter trades with DCC
        Confirmation or 1 big candle compared to previous 10 candles.
        5. Don't take trades if the 200EMA of 5Min or 15Min timeframe is near to our entry point showing the RiskReward Ratio (RRR) below 1:1.5.
        6. Aim for a minimum Risk-Reward Ratio of 1:1.5 or a logical target, Aim for 1:2 if no price action level nearby, Max Stoploss: 0.30%.
        7. Double Tops/Bottoms & Head & Shoulders Should only form on the EMA.
        8. We will avoid entry if the entry candle is bigger than 0.25%.
        9. Minimum 7-8 Candles required in both the shoulders in case of H&S, INV H&S.
        10. We will avoid entry if the recent high or low is within 50% of our overall target.


        Format your response as a structured plan with clear rules and guidelines.
        """

        try:
            # Query Gemini AI
            response = model.generate_content(prompt)
            risk_plan = response.text
            risk_plan = self.clean_gemini_response(risk_plan)
            # Store the risk management plan
            self.risk_management['overall_plan'] = risk_plan

            return risk_plan

        except Exception as e:
            print(f"Error generating risk management plan with Gemini AI: {e}")
            return f"Error: {e}"


    def plot_chart(self, symbol):
        """Plot a chart with technical indicators for the given symbol"""
        if symbol not in self.data or self.data[symbol] is None:
            print(f"No data available for {symbol}")
            return

        df = self.data[symbol].copy()

        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Main chart with price and moving averages
        ax1.plot(df['timestamp'], df['close'], label='Close Price', linewidth=2)
        ax1.plot(df['timestamp'], df['SMA_20'], label='SMA 20', linestyle='--', alpha=0.8)
        ax1.plot(df['timestamp'], df['SMA_50'], label='SMA 50', linestyle='--', alpha=0.8)
        ax1.plot(df['timestamp'], df['BB_Upper'], 'k--', alpha=0.3)
        ax1.plot(df['timestamp'], df['BB_Lower'], 'k--', alpha=0.3)
        ax1.fill_between(df['timestamp'], df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')

        # Add buy/sell annotations if available
        if symbol in self.recommendations:
            rec = self.recommendations[symbol]

            # Only plot if we have valid recommendation data
            if all(key in rec for key in ['recommendation', 'entry_price']) and rec['entry_price'] is not None:
                try:
                    last_date = df['timestamp'].iloc[-1]
                    price = float(rec['entry_price'])
                    action = rec['recommendation']
                    color = 'green' if action == "BUY" else 'red' if action == "SELL" else 'gray'

                    ax1.annotate(f"{action} @ {price:.4f}",
                                xy=(last_date, price),
                                xytext=(10, 10),
                                textcoords="offset points",
                                arrowprops=dict(arrowstyle="->", color=color),
                                color=color,
                                fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

                    # Add stop loss line if available and valid
                    if 'stop_loss' in rec and rec['stop_loss'] is not None:
                        try:
                            sl_price = float(rec['stop_loss'])
                            ax1.axhline(y=sl_price, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                            ax1.annotate(f"SL: {sl_price:.4f}",
                                        xy=(last_date, sl_price),
                                        xytext=(-80, -15),
                                        textcoords="offset points",
                                        color='red',
                                        fontsize=9)
                        except (ValueError, TypeError):
                            print(f"Invalid stop loss value for {symbol}: {rec['stop_loss']}")

                    # Add take profit line if available and valid
                    if 'take_profit' in rec and rec['take_profit'] is not None:
                        try:
                            tp_price = float(rec['take_profit'])
                            ax1.axhline(y=tp_price, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
                            ax1.annotate(f"TP: {tp_price:.4f}",
                                        xy=(last_date, tp_price),
                                        xytext=(-80, 10),
                                        textcoords="offset points",
                                        color='green',
                                        fontsize=9)
                        except (ValueError, TypeError):
                            print(f"Invalid take profit value for {symbol}: {rec['take_profit']}")

                except (ValueError, TypeError) as e:
                    print(f"Error plotting recommendation for {symbol}: {e}")

        ax1.set_title(f'{symbol} Price Chart with Technical Indicators', fontsize=14, pad=20)
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # MACD subplot
        ax2.plot(df['timestamp'], df['MACD'], label='MACD', color='blue', linewidth=1)
        ax2.plot(df['timestamp'], df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1)
        ax2.bar(df['timestamp'], df['MACD_Hist'], alpha=0.5, label='Histogram', color=np.where(df['MACD_Hist'] >= 0, 'green', 'red'))
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('MACD', fontsize=10)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # RSI subplot
        ax3.plot(df['timestamp'], df['RSI'], label='RSI', color='purple', linewidth=1)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.3)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.3)
        ax3.fill_between(df['timestamp'], 70, 30, alpha=0.1, color='gray')
        ax3.set_ylabel('RSI', fontsize=10)
        ax3.set_xlabel('Date', fontsize=10)
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save chart to static/images directory for the web app
        chart_path = os.path.join('static', 'images', f"{symbol}_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Chart for {symbol} has been saved as {chart_path}")

    def run_full_analysis(self):
        """Run a full analysis on all currency pairs"""
        # Fetch data for all currency pairs
        for pair in config.CURRENCY_PAIRS:
            self.fetch_forex_data(pair)
            time.sleep(config.API_RATE_LIMIT_DELAY)  # Avoid API rate limits

        # Analyze patterns for all pairs
        for pair in config.CURRENCY_PAIRS:
            if pair in self.data and self.data[pair] is not None:
                print(f"\nAnalyzing patterns for {pair}...")
                analysis = self.identify_patterns(pair)
                if analysis:
                    print(f"Analysis for {pair} completed.")
                time.sleep(config.API_RATE_LIMIT_DELAY)  # Avoid API rate limits

        # Generate risk management plan
        print("\nGenerating risk management plan...")
        risk_plan = self.generate_risk_management_plan()
        if risk_plan:
            print("Risk management plan generated.")

        # Generate charts
        for pair in config.CURRENCY_PAIRS:
            if pair in self.data and self.data[pair] is not None:
                self.plot_chart(pair)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print a summary of all analysis and recommendations"""
        print("\n" + "="*80)
        print("FOREX TRADING STRATEGY SUMMARY")
        print("="*80)

        for pair in config.CURRENCY_PAIRS:
            if pair in self.recommendations:
                rec = self.recommendations[pair]
                print(f"\n{pair} RECOMMENDATION:")
                print(f"Action: {rec.get('recommendation', 'N/A')}")
                print(f"Strategy: {rec.get('strategy', 'N/A')}")
                print(f"Entry Price: {rec.get('entry_price', 'N/A')}")
                print(f"Stop Loss: {rec.get('stop_loss', 'N/A')}")
                print(f"Take Profit: {rec.get('take_profit', 'N/A')}")
                print(f"Risk-Reward Ratio: {rec.get('risk_reward_ratio', 'N/A')}")
                print(f"Position Size: {rec.get('position_sizing_percentage', 'N/A')}%")
                print(f"Timeframe: {rec.get('timeframe', 'N/A')}")
                print(f"Confidence: {rec.get('confidence', 'N/A')}")

        print("\n" + "="*80)
        print("RISK MANAGEMENT PLAN SUMMARY")
        print("="*80)
        if 'overall_plan' in self.risk_management:
            print(self.risk_management['overall_plan'])
        else:
            print("No risk management plan available.")


# If the script is run directly (not imported), run the analysis
if __name__ == "__main__":
    trader = ForexTrader()
    trader.run_full_analysis()