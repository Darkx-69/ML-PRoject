# app.py

import pandas as pd
import joblib
import numpy as np
import json
from flask import Flask, request, render_template, url_for, jsonify # Import render_template and jsonify
# REMOVED: from pyngrok import ngrok
import os

# REMOVED: The try/except block for ngrok.set_auth_token

app = Flask(__name__) # Flask automatically serves files from the 'static' folder

# --- Load Models and Data (Keep this section as is) ---
try:
    # Load BOTH models
    final_pipeline = joblib.load('category_based_predictor.pkl')
    advanced_pipeline = joblib.load('advanced_predictor.pkl')

    df = pd.read_pickle('processed_data.pkl')
    with open('data_descriptions.json', 'r') as f:
        cluster_descriptions = json.load(f)
    with open('avg_lot_areas.json', 'r') as f:
        avg_lot_areas = json.load(f)
    print("Pre-trained models and data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model or data files: {e}")
    # In a production environment, you might want more robust error handling
    # For now, exiting might be okay during setup if files are missing.
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading files: {e}")
    exit()
# --- End Load Models ---


@app.route('/')
def home():
    """Renders the main page using an HTML file."""
    global df, cluster_descriptions

    # Standard data for dropdowns
    try:
        neighborhoods = sorted(df['Neighborhood'].unique().tolist())
        clusters = sorted(df['Cluster'].unique().tolist())
        overall_qual_list = sorted(df['OverallQual'].unique().tolist())
    except Exception as e:
        print(f"Error accessing DataFrame columns for dropdowns: {e}")
        # Provide default empty lists or handle error appropriately
        neighborhoods = []
        clusters = []
        overall_qual_list = []


    data_options = {
        'neighborhoods': neighborhoods,
        'clusters': clusters,
        'overall_qual_list': overall_qual_list
    }

    # Refined descriptions for Overall Quality
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

    # Pass all descriptions to the template
    data_descriptions = {
        'Cluster': cluster_descriptions,
        'Quality': quality_descriptions
    }

    # Use render_template to load index.html from the 'templates' folder
    return render_template('index.html',
                           data_options=data_options,
                           data_descriptions=data_descriptions)

# --- Predict Routes (Keep these as they are, using jsonify) ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handles the (SIMPLE) price prediction request from the user."""
    global final_pipeline, df, avg_lot_areas
    try:
        data = request.json
        lot_area_input = data.get('LotArea')

        cluster_val = data['Cluster']
        overall_qual_val = data['OverallQual']

        neighborhood_cat = f"{data['Neighborhood']}_{cluster_val}"
        user_neighborhood = data['Neighborhood']

        if lot_area_input and str(lot_area_input).isdigit(): # Check if it's a digit string
            lot_area = int(lot_area_input)
        else:
            # Use .get with a fallback to the global mean if category specific mean isn't found
            lot_area = avg_lot_areas.get(neighborhood_cat, df['LotArea'].mean())

        lot_area = float(lot_area) # Ensure float

        # --- Similar House Logic ---
        similar_house_df = df[
            (df['Neighborhood'] == user_neighborhood) &
            (df['Cluster'] == int(cluster_val))
        ].copy()

        actual_house_details = { 'found': False }

        if not similar_house_df.empty:
            # Ensure LotArea is numeric before calculating diff
            similar_house_df['LotArea'] = pd.to_numeric(similar_house_df['LotArea'], errors='coerce')
            similar_house_df.dropna(subset=['LotArea'], inplace=True) # Drop rows where LotArea became NaN

            if not similar_house_df.empty: # Check again after potential drops
                 similar_house_df['lot_area_diff'] = np.abs(similar_house_df['LotArea'] - lot_area)
                 closest_house_row = similar_house_df.sort_values('lot_area_diff').iloc[0]

                 actual_house_details = {
                    'found': True,
                    'price': f"${closest_house_row['SalePrice']:,.0f}" if pd.notna(closest_house_row['SalePrice']) else "N/A",
                    'lot_area': f"{closest_house_row['LotArea']:,.0f} sqft" if pd.notna(closest_house_row['LotArea']) else "N/A",
                    'quality': f"{int(closest_house_row['OverallQual'])}/10" if pd.notna(closest_house_row['OverallQual']) else "N/A",
                    'neighborhood': closest_house_row['Neighborhood'] or "N/A",
                    'cluster': f"Category {int(closest_house_row['Cluster'])}" if pd.notna(closest_house_row['Cluster']) else "N/A"
                 }


        # --- Average Category Price Logic ---
        category_data = df[df['Neighborhood_and_Category'] == neighborhood_cat]
        avg_category_price = 0
        if not category_data.empty and 'SalePrice' in category_data.columns:
             # Ensure SalePrice is numeric before calculating mean
             valid_prices = pd.to_numeric(category_data['SalePrice'], errors='coerce').dropna()
             if not valid_prices.empty:
                  avg_category_price = valid_prices.mean()


        # --- Size Comparison Logic ---
        lot_tolerance = 0.20
        lower_bound = lot_area * (1 - lot_tolerance)
        upper_bound = lot_area * (1 + lot_tolerance)

        # Ensure LotArea is numeric before comparison
        df_numeric_lot = df.copy()
        df_numeric_lot['LotArea'] = pd.to_numeric(df_numeric_lot['LotArea'], errors='coerce')
        df_numeric_lot.dropna(subset=['LotArea'], inplace=True)

        size_comparison_df = df_numeric_lot[(df_numeric_lot['LotArea'] >= lower_bound) & (df_numeric_lot['LotArea'] <= upper_bound)]
        size_comparison_details = {'found': False}

        if not size_comparison_df.empty:
            # Ensure SalePrice is numeric before grouping/mean
            size_comparison_df['SalePrice'] = pd.to_numeric(size_comparison_df['SalePrice'], errors='coerce')
            valid_comparison_prices = size_comparison_df.dropna(subset=['SalePrice'])

            if not valid_comparison_prices.empty:
                 neighborhood_comparison_stats = valid_comparison_prices.groupby('Neighborhood')['SalePrice'].mean()
                 if user_neighborhood in neighborhood_comparison_stats:
                     neighborhood_comparison_stats = neighborhood_comparison_stats.drop(user_neighborhood, errors='ignore') # Ignore error if not present

                 if not neighborhood_comparison_stats.empty and len(neighborhood_comparison_stats) >= 1:
                     highest_neighborhood = neighborhood_comparison_stats.idxmax()
                     highest_price = neighborhood_comparison_stats.max()
                     lowest_neighborhood = neighborhood_comparison_stats.idxmin()
                     lowest_price = neighborhood_comparison_stats.min()

                     size_comparison_details = {
                         'found': True,
                         'target_lot_area': f"{int(lot_area):,.0f} sq. ft.",
                         'highest_neighborhood': highest_neighborhood or "N/A",
                         'highest_price': f"${highest_price:,.0f}" if pd.notna(highest_price) else "N/A",
                         'lowest_neighborhood': lowest_neighborhood or "N/A",
                         'lowest_price': f"${lowest_price:,.0f}" if pd.notna(lowest_price) else "N/A"
                     }


        # --- Prediction Logic ---
        input_df = pd.DataFrame([{
            'Neighborhood_and_Category': neighborhood_cat,
            'LotArea': lot_area,
            'OverallQual': overall_qual_val
        }])
        predicted_price = final_pipeline.predict(input_df)[0]
        predicted_price = max(0, predicted_price) # Ensure non-negative


        # --- Price Comparison Logic ---
        run_price_comp = data.get('run_price_comparison', False)
        price_comparison_details = {'run': run_price_comp, 'found': False}

        if run_price_comp:
            price_tolerance = 0.10
            price_lower = predicted_price * (1 - price_tolerance)
            price_upper = predicted_price * (1 + price_tolerance)

            # Ensure SalePrice is numeric
            df_numeric_sale = df.copy()
            df_numeric_sale['SalePrice'] = pd.to_numeric(df_numeric_sale['SalePrice'], errors='coerce')
            df_numeric_sale.dropna(subset=['SalePrice'], inplace=True)


            price_comp_df = df_numeric_sale[
                (df_numeric_sale['SalePrice'] >= price_lower) &
                (df_numeric_sale['SalePrice'] <= price_upper) &
                (df_numeric_sale['Neighborhood'] != user_neighborhood)
            ]

            if not price_comp_df.empty:
                # Ensure columns for aggregation are numeric
                agg_cols_numeric = ['OverallQual', 'GrLivArea', 'LotArea']
                for col in agg_cols_numeric:
                     price_comp_df[col] = pd.to_numeric(price_comp_df[col], errors='coerce')

                # Group and aggregate, handling potential NaNs after coercion
                neighborhood_price_stats = price_comp_df.groupby('Neighborhood').agg(
                    AvgQual=('OverallQual', 'mean'),
                    AvgGrLivArea=('GrLivArea', 'mean'),
                    AvgLotArea=('LotArea', 'mean'),
                    Count = ('SalePrice', 'count') # Add count to filter neighborhoods with few samples
                ).dropna() # Drop neighborhoods where mean calculation failed

                # Filter for neighborhoods with at least, say, 3 houses in the price range
                neighborhood_price_stats = neighborhood_price_stats[neighborhood_price_stats['Count'] >= 3]

                if not neighborhood_price_stats.empty:
                     neighborhood_price_stats = neighborhood_price_stats.sort_values(by='AvgQual', ascending=False)
                     neighborhood_list = []
                     for index, row in neighborhood_price_stats.head(5).iterrows():
                         neighborhood_list.append({
                             'neighborhood': index or "N/A",
                             'quality': f"{row['AvgQual']:.1f}/10" if pd.notna(row['AvgQual']) else "N/A",
                             'liv_area': f"{row['AvgGrLivArea']:,.0f} sqft" if pd.notna(row['AvgGrLivArea']) else "N/A",
                             'lot_area': f"{row['AvgLotArea']:,.0f} sqft" if pd.notna(row['AvgLotArea']) else "N/A"
                         })

                     if neighborhood_list:
                         price_comparison_details = {
                             'run': True,
                             'found': True,
                             'target_price': f"${predicted_price:,.0f}",
                             'neighborhoods': neighborhood_list
                         }

        # --- Return Final JSON ---
        return jsonify({
            'predicted_price': f"${predicted_price:,.0f}",
            'lot_area_used': f"{int(lot_area):,.0f} sq. ft.",
            'actual_house': actual_house_details,
            'avg_price': f"${avg_category_price:,.0f}" if avg_category_price > 0 else "N/A",
            'size_comparison': size_comparison_details,
            'price_comparison': price_comparison_details
        })

    except Exception as e:
        print(f"Error in /predict: {e}")
        # Return a JSON error response
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500


@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    """Handles the (ADVANCED) price prediction request."""
    global advanced_pipeline
    try:
        data = request.json
        # Basic validation
        required_fields = ['Neighborhood', 'LotArea', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'TotalBath', 'AgeSold']
        if not all(field in data for field in required_fields):
             return jsonify({'error': 'Missing required fields for advanced prediction.'}), 400

        input_data = {
            'Neighborhood': data['Neighborhood'],
            'LotArea': float(data['LotArea']),
            'OverallQual': int(data['OverallQual']),
            'GrLivArea': float(data['GrLivArea']),
            'TotalBsmtSF': float(data['TotalBsmtSF']),
            'TotalBath': float(data['TotalBath']),
            'AgeSold': float(data['AgeSold'])
        }
        input_df = pd.DataFrame([input_data])

        # Predict (remembering the model expects log-transformed features and predicts log-transformed price)
        # We need to apply the same log transform to input features as done during training
        input_df_processed = input_df.copy()
        skewed_features_list = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
        for feature in skewed_features_list:
             if feature in input_df_processed.columns:
                  # Ensure positivity before log transform
                  if (input_df_processed[feature] >= 0).all():
                      input_df_processed[feature] = np.log1p(input_df_processed[feature])
                  else:
                      # Handle potential negative input if necessary, maybe return error or default
                       print(f"Warning: Input feature '{feature}' is negative. Using 0 for log1p.")
                       input_df_processed[feature] = 0 # Or handle as error


        predicted_price_transformed = advanced_pipeline.predict(input_df_processed)[0]

        # Reverse log transform if target was transformed during training
        # Assuming target was log-transformed (as in the corrected train_model.py)
        predicted_price = np.expm1(predicted_price_transformed)
        predicted_price = max(0, predicted_price) # Ensure non-negative

        return jsonify({
            'predicted_price': f"${predicted_price:,.0f}"
        })
    except ValueError as ve:
         print(f"ValueError in advanced prediction: {ve}")
         return jsonify({'error': 'Invalid input data type. Please check numeric fields.'}), 400
    except Exception as e:
        print(f"Error in advanced prediction: {e}")
        return jsonify({'error': 'An internal error occurred during advanced prediction.'}), 500


# --- Main Execution Block (for local testing) ---
# This part is ignored by PythonAnywhere's WSGI server
if __name__ == '__main__':
    # Use PORT environment variable if available (common in deployment platforms)
    # Default to 5001 for local testing to avoid potential conflicts with 5000
    port = int(os.environ.get("PORT", 5001))
    print(f"--- Flask app running locally on http://127.0.0.1:{port} ---")
    # Set debug=False for anything resembling production, True only for active development
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(host='0.0.0.0', port=port, debug=False)
