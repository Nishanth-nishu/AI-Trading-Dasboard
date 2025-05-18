from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
from datetime import datetime, timedelta
from forex_trader import ForexTrader
import config

app = Flask(__name__)

# Initialize the forex trader
trader = ForexTrader()

# Context processor to make 'now' available in all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

@app.route('/')
def index():
    """Main dashboard page"""
    # Get all available currency pairs
    currency_pairs = config.CURRENCY_PAIRS
    return render_template('index.html', currency_pairs=currency_pairs)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze selected currency pairs"""
    selected_pairs = request.form.getlist('currency_pairs')
    
    if not selected_pairs:
        selected_pairs = config.CURRENCY_PAIRS[:2]  # Analyze at least two pairs by default
    
    # Clear existing data to avoid stale results
    trader.data = {}
    trader.analysis = {}
    trader.recommendations = {}
    
    # Run analysis for selected pairs
    for pair in selected_pairs:
        trader.fetch_forex_data(pair)
        trader.identify_patterns(pair)
        trader.plot_chart(pair)
    
    # Generate risk management plan
    trader.generate_risk_management_plan()
    
    # Save recommendations to JSON for the frontend
    recommendations_json = {}
    for pair in selected_pairs:
        if pair in trader.recommendations:
            recommendations_json[pair] = trader.recommendations[pair]
    
    with open('static/data/recommendations.json', 'w') as f:
        json.dump(recommendations_json, f)
    
    return redirect(url_for('analysis'))

@app.route('/analysis')
def analysis():
    """Display analysis results"""
    # Check if we have recommendations
    try:
        with open('static/data/recommendations.json', 'r') as f:
            recommendations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        recommendations = {}
    
    # Load analysis texts
    analysis_texts = trader.analysis
    
    # Get list of generated charts
    chart_files = []
    for pair in trader.data.keys():
        chart_path = f"images/{pair}_chart.png"
        if os.path.exists(f"static/{chart_path}"):
            chart_files.append({"pair": pair, "path": chart_path})
    
    return render_template('analysis.html', 
                          recommendations=recommendations, 
                          analysis_texts=analysis_texts,
                          chart_files=chart_files)

@app.route('/risk')
def risk():
    """Display risk management plan"""
    risk_plan = trader.risk_management.get('overall_plan', 'No risk management plan available.')
    
    # Check if we have recommendations
    try:
        with open('static/data/recommendations.json', 'r') as f:
            recommendations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        recommendations = {}
    
    return render_template('risk.html', risk_plan=risk_plan, recommendations=recommendations)

@app.route('/api/data/<currency_pair>')
def get_data(currency_pair):
    """API endpoint to get data for a specific currency pair"""
    if currency_pair in trader.data:
        # Convert DataFrame to JSON
        data_json = trader.data[currency_pair].to_json(orient='records', date_format='iso')
        return data_json
    else:
        return jsonify({"error": "Currency pair data not found"}), 404

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('static/data', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    app.run(debug=True, port=5000)