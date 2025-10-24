# app.py

import pandas as pd
import joblib
import numpy as np
import json
from flask import Flask, request, render_template, url_for, jsonify
# from pyngrok import ngrok # Removed for deployment
import os

app = Flask(__name__)

# --- Load Models and Data ---
try:
    # Use relative paths, assuming files are in the same directory as app.py
    # Deployment platforms might place files differently, adjust if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_pipeline = joblib.load(os.path.join(current_dir,'category_based_predictor.pkl'))
    advanced_pipeline = joblib.load(os.path.join(current_dir,'advanced_predictor.pkl'))
    df = pd.read_pickle(os.path.join(current_dir,'processed_data.pkl'))

    # Convert relevant columns to numeric ONCE after loading, coercing errors
    numeric_cols_to_check = ['LotArea', 'SalePrice', 'OverallQual', 'Cluster', 'GrLivArea', 'TotalBsmtSF']
    for col in numeric_cols_to_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
             print(f"Warning: Expected column '{col}' not found in processed_data.pkl")

    # Drop rows where essential numeric cols became NaN after coercion, or where key categoricals are missing
    df.dropna(subset=['LotArea', 'SalePrice', 'OverallQual', 'Cluster', 'Neighborhood'], inplace=True)
    # Ensure Cluster is integer after potential coercion/dropna
    if 'Cluster' in df.columns:
        df['Cluster'] = df['Cluster'].astype(int)
    if 'Neighborhood_and_Category' in df.columns:
         df['Neighborhood_and_Category'] = df['Neighborhood_and_Category'].astype(str)


    with open(os.path.join(current_dir,'data_descriptions.json'), 'r') as f:
        cluster_descriptions = json.load(f)
    with open(os.path.join(current_dir,'avg_lot_areas.json'), 'r') as f:
        avg_lot_areas = json.load(f)

    print(f"Dataframe shape after loading and cleaning: {df.shape}")
    print("Pre-trained models and data loaded successfully.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Model or data file not found: {e}. Ensure .pkl and .json files are in the deployment package.")
    # In a real app, you might return an error page or raise an exception
    exit()
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred loading files: {e}")
    exit()
# --- End Load Models ---


@app.route('/')
def home():
    """Renders the main page using an HTML file."""
    global df, cluster_descriptions

    # Standard data for dropdowns
    try:
        # Use try-except as df might be empty if loading failed somehow (though we exit above)
        neighborhoods = sorted(df['Neighborhood'].unique().tolist()) if 'Neighborhood' in df.columns else []
        # Ensure Cluster exists and contains numeric data before getting unique values
        clusters = sorted(df['Cluster'].dropna().astype(int).unique().tolist()) if 'Cluster' in df.columns else []
        overall_qual_list = sorted(df['OverallQual'].dropna().astype(int).unique().tolist()) if 'OverallQual' in df.columns else []
    except Exception as e:
        print(f"Error accessing DataFrame columns for dropdowns: {e}")
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
    data_descriptions_for_template = {
        'Cluster': cluster_descriptions,
        'Quality': quality_descriptions
    }

    try:
        return render_template('index.html',
                               data_options=data_options,
                               data_descriptions=data_descriptions_for_template)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "Error loading page.", 500


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the (SIMPLE) price prediction request with robust error handling."""
    global final_pipeline, df, avg_lot_areas
    # --- Default values ---
    predicted_price_str = "N/A"
    lot_area_used_str = "N/A"
    actual_house_details = { 'found': False }
    avg_category_price_str = "N/A"
    size_comparison_details = {'found': False}
    price_comparison_details = {'run': False, 'found': False}

    try:
        data = request.json
        print(f"Received simple prediction request data: {data}") # DEBUG LOG

        # --- Input Validation ---
        required_simple = ['Neighborhood', 'Cluster', 'OverallQual']
        if not all(field in data for field in required_simple):
             print("Error: Missing required fields in simple prediction request.")
             raise ValueError("Missing required fields.")

        lot_area_input = data.get('LotArea') # Optional field
        user_neighborhood = data['Neighborhood']
        try:
             cluster_val = int(data['Cluster'])
             overall_qual_val = int(data['OverallQual'])
        except (ValueError, TypeError):
             print("Error: Cluster or OverallQual is not a valid integer.")
             raise ValueError("Invalid Cluster or OverallQual.")

        # Construct category key BEFORE calculating lot_area
        neighborhood_cat = f"{user_neighborhood}_{cluster_val}"
        print(f"Constructed neighborhood_cat: {neighborhood_cat}") # DEBUG LOG

        # --- Determine Lot Area ---
        if lot_area_input and str(lot_area_input).replace('.', '', 1).isdigit(): # Allow floats too
            lot_area = float(lot_area_input)
            if lot_area <= 0:
                print("Warning: Received non-positive LotArea input, using default.")
                lot_area = avg_lot_areas.get(neighborhood_cat, df['LotArea'].mean())
        else:
            lot_area = avg_lot_areas.get(neighborhood_cat, df['LotArea'].mean())
            print(f"LotArea input invalid or missing. Using default/average: {lot_area}") # DEBUG LOG
        
        lot_area = float(lot_area) # Ensure float
        lot_area_used_str = f"{int(lot_area):,.0f} sq. ft."
        print(f"Final LotArea used for prediction: {lot_area}") # DEBUG LOG

        # --- Similar House Logic ---
        try:
            # Ensure df has required columns and correct types (done mostly at load time now)
            similar_house_df = df[
                (df['Neighborhood'] == user_neighborhood) &
                (df['Cluster'] == cluster_val) # Already int
            ].copy()
            print(f"Found {len(similar_house_df)} houses in {user_neighborhood}, Category {cluster_val}") # DEBUG LOG

            if not similar_house_df.empty:
                 # Calculate difference using the numeric LotArea column
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
                 print(f"Found closest similar house: {actual_house_details}") # DEBUG LOG
            else:
                 print("No similar houses found for category/neighborhood combo.") # DEBUG LOG
        except Exception as e:
            print(f"Error during Similar House lookup: {e}") # Log error but continue

        # --- Average Category Price Logic ---
        try:
            category_data = df[df['Neighborhood_and_Category'] == neighborhood_cat]
            if not category_data.empty:
                # Use the already cleaned/numeric SalePrice column
                valid_prices = category_data['SalePrice'].dropna()
                if not valid_prices.empty:
                    avg_category_price = valid_prices.mean()
                    avg_category_price_str = f"${avg_category_price:,.0f}"
                    print(f"Avg price for {neighborhood_cat}: {avg_category_price_str}") # DEBUG LOG
                else:
                    print(f"No valid prices found for category {neighborhood_cat} to calculate average.") # DEBUG LOG
            else:
                 print(f"No data found for category {neighborhood_cat} to calculate average.") # DEBUG LOG
        except Exception as e:
            print(f"Error calculating Average Category Price: {e}")

        # --- Size Comparison Logic ---
        try:
            lot_tolerance = 0.20
            lower_bound = lot_area * (1 - lot_tolerance)
            upper_bound = lot_area * (1 + lot_tolerance)

            # Use the already cleaned/numeric LotArea column
            size_comparison_df = df[(df['LotArea'] >= lower_bound) & (df['LotArea'] <= upper_bound)]
            print(f"Found {len(size_comparison_df)} houses with LotArea between {lower_bound:.0f} and {upper_bound:.0f}") # DEBUG LOG

            if not size_comparison_df.empty:
                # Use cleaned SalePrice
                valid_comparison_prices = size_comparison_df.dropna(subset=['SalePrice'])
                if not valid_comparison_prices.empty:
                     neighborhood_comparison_stats = valid_comparison_prices.groupby('Neighborhood')['SalePrice'].mean()
                     if user_neighborhood in neighborhood_comparison_stats:
                         neighborhood_comparison_stats = neighborhood_comparison_stats.drop(user_neighborhood, errors='ignore')

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
                         print(f"Size comparison stats: {size_comparison_details}") # DEBUG LOG
                     else:
                          print("Not enough other neighborhoods found for size comparison.") # DEBUG LOG
                else:
                     print("No valid prices in size comparison group.") # DEBUG LOG
            else:
                 print("No houses found within lot area bounds for size comparison.") # DEBUG LOG
        except Exception as e:
            print(f"Error during Size Comparison: {e}")


        # --- Prediction Logic ---
        try:
            input_df = pd.DataFrame([{
                'Neighborhood_and_Category': neighborhood_cat,
                'LotArea': lot_area, # Already float
                'OverallQual': overall_qual_val # Already int
            }])
            print(f"Input DataFrame for simple prediction: {input_df.to_dict()}") # DEBUG LOG

            predicted_price = final_pipeline.predict(input_df)[0]
            predicted_price = max(0, float(predicted_price)) # Ensure non-negative float
            predicted_price_str = f"${predicted_price:,.0f}"
            print(f"Simple model predicted price: {predicted_price_str}") # DEBUG LOG
        except Exception as e:
            print(f"Error during Simple Model prediction: {e}")
            predicted_price = 0 # Set default to allow price comparison logic


        # --- Price Comparison Logic ---
        run_price_comp = data.get('run_price_comparison', False)
        price_comparison_details['run'] = run_price_comp # Update run status

        if run_price_comp and predicted_price > 0:
            try:
                price_tolerance = 0.10
                price_lower = predicted_price * (1 - price_tolerance)
                price_upper = predicted_price * (1 + price_tolerance)

                # Use cleaned SalePrice column
                price_comp_df = df[
                    (df['SalePrice'] >= price_lower) &
                    (df['SalePrice'] <= price_upper) &
                    (df['Neighborhood'] != user_neighborhood)
                ]
                print(f"Found {len(price_comp_df)} houses for price comparison around ${predicted_price:,.0f}") # DEBUG LOG


                if not price_comp_df.empty:
                    # Use cleaned numeric columns
                    agg_cols_numeric = ['OverallQual', 'GrLivArea', 'LotArea']
                    valid_price_comp_df = price_comp_df.dropna(subset=agg_cols_numeric) # Ensure numeric cols are not NaN

                    if not valid_price_comp_df.empty:
                         neighborhood_price_stats = valid_price_comp_df.groupby('Neighborhood').agg(
                             AvgQual=('OverallQual', 'mean'),
                             AvgGrLivArea=('GrLivArea', 'mean'),
                             AvgLotArea=('LotArea', 'mean'),
                             Count = ('SalePrice', 'count')
                         ).dropna()

                         neighborhood_price_stats = neighborhood_price_stats[neighborhood_price_stats['Count'] >= 3] # Filter low counts

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
                                 price_comparison_details.update({
                                     'found': True,
                                     'target_price': f"${predicted_price:,.0f}",
                                     'neighborhoods': neighborhood_list
                                 })
                                 print("Generated price comparison details.") # DEBUG LOG
                             else:
                                 print("Price comparison grouping resulted in empty list after filtering.") # DEBUG LOG
                         else:
                             print("No neighborhoods met count criteria for price comparison.") # DEBUG LOG
                    else:
                         print("No valid rows after dropping NaNs for price comparison aggregation.") # DEBUG LOG
                else:
                     print("No houses found in price range for comparison.") # DEBUG LOG
            except Exception as e:
                print(f"Error during Price Comparison: {e}")
        elif run_price_comp:
             print("Skipping price comparison due to zero predicted price.") # DEBUG LOG


    except ValueError as ve:
         print(f"ValueError in /predict: {ve}")
         # Return mostly defaults but indicate error source if possible
         return jsonify({'error': str(ve), 'predicted_price': predicted_price_str, 'lot_area_used': lot_area_used_str, 'actual_house': actual_house_details, 'avg_price': avg_category_price_str, 'size_comparison': size_comparison_details, 'price_comparison': price_comparison_details}), 400
    except Exception as e:
        print(f"General Error in /predict: {e}")
        # Return mostly defaults
        return jsonify({'error': 'An internal error occurred.', 'predicted_price': predicted_price_str, 'lot_area_used': lot_area_used_str, 'actual_house': actual_house_details, 'avg_price': avg_category_price_str, 'size_comparison': size_comparison_details, 'price_comparison': price_comparison_details}), 500


    # --- Return Final JSON ---
    return jsonify({
        'predicted_price': predicted_price_str,
        'lot_area_used': lot_area_used_str,
        'actual_house': actual_house_details,
        'avg_price': avg_category_price_str,
        'size_comparison': size_comparison_details,
        'price_comparison': price_comparison_details
    })


@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    """Handles the (ADVANCED) price prediction request with error handling."""
    global advanced_pipeline
    predicted_price_str = "N/A" # Default

    try:
        data = request.json
        print(f"Received advanced prediction request data: {data}") # DEBUG LOG

        # --- Input Validation ---
        required_fields = ['Neighborhood', 'LotArea', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'TotalBath', 'AgeSold']
        if not all(field in data and data[field] is not None for field in required_fields):
             print(f"Error: Missing or null required fields for advanced prediction. Received: {data}")
             raise ValueError("Missing required fields for advanced prediction.")

        # --- Data Conversion and Basic Checks ---
        input_data = {}
        numeric_fields_adv = ['LotArea', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'TotalBath', 'AgeSold']
        for field in required_fields:
            if field == 'Neighborhood':
                input_data[field] = str(data[field]) # Ensure string
            else:
                 try:
                      val = float(data[field])
                      if field in ['OverallQual', 'AgeSold'] and val < 0:
                           raise ValueError(f"{field} cannot be negative.")
                      if field in ['LotArea', 'GrLivArea', 'TotalBsmtSF'] and val < 0:
                           # Allow 0 for basement, but not others ideally
                           if field == 'TotalBsmtSF' and val == 0:
                                input_data[field] = 0.0
                           else:
                                raise ValueError(f"{field} must be non-negative.")
                      elif field == 'TotalBath' and val < 0:
                           raise ValueError(f"{field} cannot be negative.")

                      input_data[field] = val
                 except (ValueError, TypeError):
                      print(f"Error: Invalid non-numeric value for field '{field}': {data[field]}")
                      raise ValueError(f"Invalid numeric value for {field}.")

        input_df = pd.DataFrame([input_data])
        print(f"Input DataFrame for advanced prediction (raw): {input_df.to_dict()}") # DEBUG LOG


        # --- Apply Preprocessing (Log Transform Features) ---
        # Mimic the transformation done during training
        input_df_processed = input_df.copy()
        skewed_features_list = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
        for feature in skewed_features_list:
             if feature in input_df_processed.columns:
                  # Apply log1p, assuming input values are already validated >= 0
                  input_df_processed[feature] = np.log1p(input_df_processed[feature])

        print(f"Input DataFrame for advanced prediction (processed): {input_df_processed.to_dict()}") # DEBUG LOG

        # --- Prediction ---
        predicted_price_transformed = advanced_pipeline.predict(input_df_processed)[0]

        # Reverse log transform (assuming target was log-transformed)
        predicted_price = np.expm1(predicted_price_transformed)
        predicted_price = max(0, float(predicted_price)) # Ensure non-negative float
        predicted_price_str = f"${predicted_price:,.0f}"
        print(f"Advanced model predicted price: {predicted_price_str}") # DEBUG LOG

        return jsonify({
            'predicted_price': predicted_price_str
        })

    except ValueError as ve:
         print(f"ValueError in /predict_advanced: {ve}")
         return jsonify({'error': str(ve), 'predicted_price': 'Error'}), 400 # Return 400 for bad input
    except Exception as e:
        print(f"General Error in /predict_advanced: {e}")
        # Log the full traceback for detailed debugging on the server
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal error occurred during prediction.', 'predicted_price': 'Error'}), 500


# --- Main Execution Block (for local testing) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"--- Flask app running locally on http://127.0.0.1:{port} ---")
    # Set debug=True ONLY for local development, NEVER in production deployment
    app.run(host='0.0.0.0', port=port, debug=True)

