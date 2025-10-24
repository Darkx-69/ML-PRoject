import os
import pandas as pd
import joblib
import numpy as np
import json
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# --- Load Models and Data ---
try:
    final_pipeline = joblib.load('category_based_predictor.pkl')
    advanced_pipeline = joblib.load('advanced_predictor.pkl')

    df = pd.read_pickle('processed_data.pkl')
    with open('data_descriptions.json', 'r') as f:
        cluster_descriptions = json.load(f)
    with open('avg_lot_areas.json', 'r') as f:
        avg_lot_areas = json.load(f)

    print("✅ Models and data loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    exit()
except Exception as e:
    print(f"❌ Error loading resources: {e}")
    exit()


@app.route('/')
def home():
    """Renders the main HTML page."""
    try:
        neighborhoods = sorted(df['Neighborhood'].unique().tolist())
        clusters = sorted(df['Cluster'].unique().tolist())
        overall_qual_list = sorted(df['OverallQual'].unique().tolist())
    except Exception as e:
        print(f"Error preparing dropdowns: {e}")
        neighborhoods, clusters, overall_qual_list = [], [], []

    data_options = {
        'neighborhoods': neighborhoods,
        'clusters': clusters,
        'overall_qual_list': overall_qual_list
    }

    quality_descriptions = {
        "1": "As-Is Condition: Requires significant renovation.",
        "2": "Foundation Potential: Needs major repairs throughout.",
        "3": "Renovation Opportunity: Requires substantial updates.",
        "4": "Basic Finishes: Functional but needs cosmetic updates.",
        "5": "Standard Finishes: Average materials, possibly dated but maintained.",
        "6": "Solid Construction: Well-maintained with decent finishes.",
        "7": "Upgraded Features: Modern finishes and good condition.",
        "8": "Quality Finishes: High-quality construction and materials.",
        "9": "Premium Materials: Custom details and high-end finishes.",
        "10": "Exceptional Quality: Luxury construction with top-of-the-line finishes."
    }

    data_descriptions = {'Cluster': cluster_descriptions, 'Quality': quality_descriptions}

    return render_template(
        'index.html',
        data_options=data_options,
        data_descriptions=data_descriptions
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        lot_area_input = data.get('LotArea')

        cluster_val = data['Cluster']
        overall_qual_val = data['OverallQual']
        user_neighborhood = data['Neighborhood']
        neighborhood_cat = f"{user_neighborhood}_{cluster_val}"

        # Handle lot area safely
        if lot_area_input and str(lot_area_input).isdigit():
            lot_area = int(lot_area_input)
        else:
            lot_area = avg_lot_areas.get(neighborhood_cat, df['LotArea'].mean())

        lot_area = float(lot_area)

        # Prediction
        input_df = pd.DataFrame([{
            'Neighborhood_and_Category': neighborhood_cat,
            'LotArea': lot_area,
            'OverallQual': overall_qual_val
        }])

        predicted_price = final_pipeline.predict(input_df)[0]
        predicted_price = max(0, predicted_price)

        return jsonify({'predicted_price': f"${predicted_price:,.0f}"})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': 'Internal error occurred'}), 500


@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    try:
        data = request.json
        required = ['Neighborhood', 'LotArea', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'TotalBath', 'AgeSold']
        if not all(k in data for k in required):
            return jsonify({'error': 'Missing required fields'}), 400

        input_df = pd.DataFrame([{
            'Neighborhood': data['Neighborhood'],
            'LotArea': float(data['LotArea']),
            'OverallQual': int(data['OverallQual']),
            'GrLivArea': float(data['GrLivArea']),
            'TotalBsmtSF': float(data['TotalBsmtSF']),
            'TotalBath': float(data['TotalBath']),
            'AgeSold': float(data['AgeSold'])
        }])

        skewed = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
        for col in skewed:
            input_df[col] = np.log1p(input_df[col].clip(lower=0))

        predicted = advanced_pipeline.predict(input_df)[0]
        predicted = np.expm1(predicted)
        predicted = max(0, predicted)

        return jsonify({'predicted_price': f"${predicted:,.0f}"})
    except Exception as e:
        print(f"Error in /predict_advanced: {e}")
        return jsonify({'error': 'Internal error occurred'}), 500


# ✅ This part ensures Render runs it correctly
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
