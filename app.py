# app.py

import pandas as pd
import joblib
import numpy as np
import json
from flask import Flask, request, render_template_string
from pyngrok import ngrok
import os


try:
    ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
    print("Ngrok token set successfully.")
except Exception as e:
    print(f"Could not set ngrok token: {e}")
    print("The app will run locally, but you won't get a public URL.")


app = Flask(__name__)

try:
    final_pipeline = joblib.load('category_based_predictor.pkl')
    df = pd.read_pickle('processed_data.pkl')
    with open('data_descriptions.json', 'r') as f:
        cluster_descriptions = json.load(f)
    with open('avg_lot_areas.json', 'r') as f:
        avg_lot_areas = json.load(f)
    print("Pre-trained model and data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model or data files: {e}")
    print("Please run the 'train_model.py' script first to create the necessary files.")
    exit()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">
        <h1 class="text-3xl font-bold mb-6 text-center text-gray-800">House Price Predictor 🏡</h1>
        <form id="prediction-form" class="space-y-6">
            <div class="space-y-4">
                <div>
                    <label for="Neighborhood" class="block text-sm font-medium text-gray-700">Neighborhood</label>
                    <select id="Neighborhood" name="Neighborhood" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                        {% for option in data_options.neighborhoods %}
                        <option value="{{ option }}">{{ option }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label for="Cluster" class="block text-sm font-medium text-gray-700">House Category</label>
                    <select id="Cluster" name="Cluster" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                        {% for option in data_options.clusters %}
                        <option value="{{ option }}">Category {{ option }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label for="OverallQual" class="block text-sm font-medium text-gray-700">Overall Quality (1-10)</label>
                    <select id="OverallQual" name="OverallQual" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2">
                        {% for option in data_options.overall_qual %}
                        <option value="{{ option }}">{{ option }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label for="LotArea" class="block text-sm font-medium text-gray-700">Lot Area (Sq. Ft.)</label>
                    <p class="text-xs text-gray-500 mt-1">Leave blank to use the average lot size for this house type.</p>
                    <input type="number" id="LotArea" name="LotArea" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2" placeholder="e.g., 10000">
                </div>
            </div>
            <div class="mt-6 text-center">
                <button type="submit" class="w-full inline-flex justify-center rounded-md border border-transparent bg-indigo-600 py-3 px-6 text-base font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition duration-300 ease-in-out">
                    Get Price Prediction
                </button>
            </div>
        </form>
        <div id="results" class="mt-8 text-center space-y-4">
            <div id="loading" class="hidden text-gray-500">
                <p>Calculating prediction...</p>
            </div>
            <div id="prediction-output" class="hidden p-4 bg-gray-50 rounded-lg">
                <h2 class="text-xl font-semibold text-gray-800">Prediction Results</h2>
                <div class="mt-2 space-y-2 text-left">
                    <p class="text-gray-600"><span class="font-medium">Predicted Price:</span> <span id="predicted-price" class="text-indigo-600 font-bold text-lg"></span></p>
                    <p class="text-gray-600"><span class="font-medium">Lot Area Used:</span> <span id="lot-area-used" class="font-medium text-base"></span></p>
                    <p class="text-gray-600"><span class="font-medium">Price of Similar House in Data:</span> <span id="actual-price" class="text-green-600 font-bold text-lg"></span></p>
                    <p class="text-gray-600"><span class="font-medium">Average Price for this Category:</span> <span id="avg-price" class="text-blue-600 font-bold text-lg"></span></p>
                </div>
            </div>
        </div>
    </div>
    <script>
        // The JavaScript that updated the description has been removed.
        document.getElementById('prediction-form').addEventListener('submit', async function(event) {
            event.preventDefault();
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('prediction-output').classList.add('hidden');
            const form = event.target;
            const data = {
                Neighborhood: form.Neighborhood.value,
                Cluster: form.Cluster.value,
                LotArea: form.LotArea.value,
                OverallQual: parseInt(form.OverallQual.value)
            };
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('prediction-output').classList.remove('hidden');
            document.getElementById('predicted-price').textContent = result.predicted_price;
            document.getElementById('lot-area-used').textContent = result.lot_area_used;
            document.getElementById('actual-price').textContent = result.actual_price;
            document.getElementById('avg-price').textContent = result.avg_price;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Renders the main page with dropdowns for user input."""
    global df, cluster_descriptions
    neighborhoods = sorted(df['Neighborhood'].unique().tolist())
    clusters = sorted(df['Cluster'].unique().tolist())
    overall_qual = sorted(df['OverallQual'].unique().tolist())
    data_options = {
        'neighborhoods': neighborhoods,
        'clusters': clusters,
        'overall_qual': overall_qual
    }
    
    data_descriptions = {
        'Cluster': cluster_descriptions
    }
    return render_template_string(HTML_TEMPLATE, data_options=data_options, data_descriptions=data_descriptions)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the price prediction request from the user."""
    global final_pipeline, df, avg_lot_areas
    data = request.json
    lot_area_input = data.get('LotArea')
    neighborhood_cat = f"{data['Neighborhood']}_{data['Cluster']}"
    
    if lot_area_input and lot_area_input.isdigit():
        lot_area = int(lot_area_input)
    else:
        
        lot_area = avg_lot_areas.get(neighborhood_cat, df['LotArea'].mean())

    
    similar_house_df = df[
        (df['Neighborhood'] == data['Neighborhood']) &
        (df['Cluster'] == int(data['Cluster']))
    ].copy()
    
    actual_price_str = "N/A for this category"
    if not similar_house_df.empty:
        
        similar_house_df['lot_area_diff'] = np.abs(similar_house_df['LotArea'] - lot_area)
        closest_house_row = similar_house_df.sort_values('lot_area_diff').iloc[0]
        actual_price = closest_house_row['SalePrice']
        actual_lot_area = closest_house_row['LotArea']
        actual_price_str = f"${actual_price:,.0f} (Lot: {actual_lot_area:,.0f} sqft)"

    
    category_data = df[df['Neighborhood_and_Category'] == neighborhood_cat]
    avg_category_price = 0
    if not category_data.empty:
        avg_category_price = category_data['SalePrice'].mean()

  
    input_df = pd.DataFrame([{
        'Neighborhood_and_Category': neighborhood_cat,
        'LotArea': lot_area,
        'OverallQual': data['OverallQual']
    }])
    predicted_price = final_pipeline.predict(input_df)[0]

    return {
        'predicted_price': f"${predicted_price:,.0f}",
        'lot_area_used': f"{int(lot_area):,.0f} sq. ft.",
        'actual_price': actual_price_str,
        'avg_price': f"${avg_category_price:,.0f}"
    }


if __name__ == '__main__':
    port = 5000
    try:
        public_url = ngrok.connect(port).public_url
        print("=" * 60)
        print(f"✅ Your app is live and accessible at this public URL: {public_url}")
        print("=" * 60)
    except Exception as e:
        print(f"Could not start ngrok. Error: {e}")
        print("You can still access the app locally at http://127.0.0.1:5000")
    
    
    app.run(port=port)