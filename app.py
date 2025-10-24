# app.py

import pandas as pd
import joblib
import numpy as np
import json
from flask import Flask, request, render_template, url_for, jsonify
# from pyngrok import ngrok # Removed for deployment
import os
import traceback # Import traceback for detailed error logging

app = Flask(__name__)

# --- Load Models and Data (Keep the robust loading from previous version) ---
try:
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

    df.dropna(subset=['LotArea', 'SalePrice', 'OverallQual', 'Cluster', 'Neighborhood'], inplace=True)
    if 'Cluster' in df.columns:
        df['Cluster'] = df['Cluster'].astype(int)
    # Ensure Neighborhood_and_Category exists before using it
    if 'Cluster' in df.columns and 'Neighborhood' in df.columns:
         df['Neighborhood_and_Category'] = df['Neighborhood'].astype(str) + '_' + df['Cluster'].astype(str)


    with open(os.path.join(current_dir,'data_descriptions.json'), 'r') as f:
        cluster_descriptions = json.load(f)
    with open(os.path.join(current_dir,'avg_lot_areas.json'), 'r') as f:
        avg_lot_areas = json.load(f)

    print(f"Dataframe shape after loading and cleaning: {df.shape}")
    print("Pre-trained models and data loaded successfully.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Model or data file not found: {e}. Ensure .pkl and .json files are in the deployment package.")
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
        neighborhoods = sorted(df['Neighborhood'].unique().tolist()) if 'Neighborhood' in df.columns else []
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

# --- /predict route (Keep the robust version from previous step) ---
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
        if not all(field in data and data[field] is not None for field in required_simple):
             print("Error: Missing or null required fields in simple prediction request.")
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
        # Ensure Neighborhood_and_Category column exists for avg_lot_areas fallback and category price lookup
        if 'Neighborhood_and_Category' not in df.columns:
             print("Error: 'Neighborhood_and_Category' column missing from DataFrame.")
             raise KeyError("'Neighborhood_and_Category' column missing")
        neighborhood_cat = f"{user_neighborhood}_{cluster_val}"
        print(f"Constructed neighborhood_cat: {neighborhood_cat}") # DEBUG LOG

        # --- Determine Lot Area ---
        if lot_area_input and str(lot_area_input).replace('.', '', 1).isdigit():
            lot_area = float(lot_area_input)
            if lot_area <= 0:
                print("Warning: Received non-positive LotArea input, using default.")
                lot_area = avg_lot_areas.get(neighborhood_cat, df['LotArea'].mean())
        else:
            lot_area = avg_lot_areas.get(neighborhood_cat, df['LotArea'].mean())
            print(f"LotArea input invalid or missing. Using default/average: {lot_area}") # DEBUG LOG

        lot_area = float(lot_area)
        lot_area_used_str = f"{int(lot_area):,.0f} sq. ft."
        print(f"Final LotArea used for prediction: {lot_area}") # DEBUG LOG

        # --- Similar House Logic ---
        try:
            similar_house_df = df[
                (df['Neighborhood'] == user_neighborhood) &
                (df['Cluster'] == cluster_val)
            ].copy()
            print(f"Found {len(similar_house_df)} houses in {user_neighborhood}, Category {cluster_val}") # DEBUG LOG

            if not similar_house_df.empty:
                 similar_house_df['lot_area_diff'] = np.abs(similar_house_df['LotArea'] - lot_area)
                 # Ensure not comparing NaN diffs
                 closest_house_row = similar_house_df.dropna(subset=['lot_area_diff']).sort_values('lot_area_diff').iloc[0]

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
        except IndexError:
             print("IndexError in Similar House lookup (likely empty after dropna).") # DEBUG LOG
             actual_house_details['found'] = False # Ensure it's false
        except Exception as e:
            print(f"Error during Similar House lookup: {e}")

        # --- Average Category Price Logic ---
        try:
            category_data = df[df['Neighborhood_and_Category'] == neighborhood_cat]
            if not category_data.empty:
                valid_prices = category_data['SalePrice'].dropna()
                if not valid_prices.empty:
                    avg_category_price = valid_prices.mean()
                    if pd.notna(avg_category_price):
                        avg_category_price_str = f"${avg_category_price:,.0f}"
                        print(f"Avg price for {neighborhood_cat}: {avg_category_price_str}") # DEBUG LOG
                    else:
                        print(f"Avg price calculation resulted in NaN for {neighborhood_cat}.") # DEBUG LOG
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

            size_comparison_df = df[(df['LotArea'] >= lower_bound) & (df['LotArea'] <= upper_bound)]
            print(f"Found {len(size_comparison_df)} houses with LotArea between {lower_bound:.0f} and {upper_bound:.0f}") # DEBUG LOG

            if not size_comparison_df.empty:
                valid_comparison_prices = size_comparison_df.dropna(subset=['SalePrice'])
                if not valid_comparison_prices.empty:
                     neighborhood_comparison_stats = valid_comparison_prices.groupby('Neighborhood')['SalePrice'].mean()
                     if user_neighborhood in neighborhood_comparison_stats:
                         neighborhood_comparison_stats = neighborhood_comparison_stats.drop(user_neighborhood, errors='ignore')

                     if not neighborhood_comparison_stats.empty and len(neighborhood_comparison_stats) >= 1:
                         # Check if max/min result in NaN before accessing idxmax/idxmin
                         if pd.notna(neighborhood_comparison_stats.max()) and pd.notna(neighborhood_comparison_stats.min()):
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
                              print("Size comparison max/min resulted in NaN.") #DEBUG LOG
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
                'LotArea': lot_area,
                'OverallQual': overall_qual_val
            }])
            print(f"Input DataFrame for simple prediction: {input_df.to_dict()}") # DEBUG LOG

            predicted_price = final_pipeline.predict(input_df)[0]
            predicted_price = max(0, float(predicted_price))
            predicted_price_str = f"${predicted_price:,.0f}"
            print(f"Simple model predicted price: {predicted_price_str}") # DEBUG LOG
        except Exception as e:
            print(f"Error during Simple Model prediction: {e}")
            predicted_price = 0 # Ensure predicted_price exists for price comparison logic


        # --- Price Comparison Logic ---
        run_price_comp = data.get('run_price_comparison', False)
        price_comparison_details['run'] = run_price_comp

        if run_price_comp and predicted_price > 0:
            try:
                price_tolerance = 0.10
                price_lower = predicted_price * (1 - price_tolerance)
                price_upper = predicted_price * (1 + price_tolerance)

                price_comp_df = df[
                    (df['SalePrice'] >= price_lower) &
                    (df['SalePrice'] <= price_upper) &
                    (df['Neighborhood'] != user_neighborhood)
                ]
                print(f"Found {len(price_comp_df)} houses for price comparison around ${predicted_price:,.0f}") # DEBUG LOG

                if not price_comp_df.empty:
                    agg_cols_numeric = ['OverallQual', 'GrLivArea', 'LotArea']
                    valid_price_comp_df = price_comp_df.dropna(subset=agg_cols_numeric)

                    if not valid_price_comp_df.empty:
                         neighborhood_price_stats = valid_price_comp_df.groupby('Neighborhood').agg(
                             AvgQual=('OverallQual', 'mean'),
                             AvgGrLivArea=('GrLivArea', 'mean'),
                             AvgLotArea=('LotArea', 'mean'),
                             Count = ('SalePrice', 'count')
                         ).dropna() # Drop rows where any mean failed

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


    except (ValueError, KeyError, TypeError) as input_error:
         print(f"Input Error in /predict: {input_error}")
         # Return mostly defaults but indicate error source
         return jsonify({'error': str(input_error), 'predicted_price': 'Error', 'lot_area_used': 'N/A', 'actual_house': actual_house_details, 'avg_price': 'N/A', 'size_comparison': size_comparison_details, 'price_comparison': price_comparison_details}), 400
    except Exception as e:
        print(f"General Error in /predict: {e}")
        traceback.print_exc() # Print full traceback to logs
        # Return mostly defaults
        return jsonify({'error': 'An internal server error occurred.', 'predicted_price': 'Error', 'lot_area_used': 'N/A', 'actual_house': actual_house_details, 'avg_price': 'N/A', 'size_comparison': size_comparison_details, 'price_comparison': price_comparison_details}), 500


    # --- Return Final JSON ---
    return jsonify({
        'predicted_price': predicted_price_str,
        'lot_area_used': lot_area_used_str,
        'actual_house': actual_house_details,
        'avg_price': avg_category_price_str,
        'size_comparison': size_comparison_details,
        'price_comparison': price_comparison_details
    })


# --- /predict_advanced route (MODIFIED with Logging & Preprocessing) ---
@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    """Handles the (ADVANCED) price prediction request with error handling and correct preprocessing."""
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
        # Define expected types for clarity
        field_types = {
            'Neighborhood': str, 'LotArea': float, 'OverallQual': int,
            'GrLivArea': float, 'TotalBsmtSF': float, 'TotalBath': float, 'AgeSold': int
        }
        for field in required_fields:
            try:
                # Attempt conversion based on expected type
                if field_types[field] == str:
                    val = str(data[field])
                elif field_types[field] == int:
                    val = int(float(data[field])) # Allow float input but store as int if expected
                else: # float
                    val = float(data[field])

                # Value range checks
                if field in ['OverallQual', 'AgeSold'] and val < 0:
                     raise ValueError(f"{field} cannot be negative.")
                # Allow 0 for basement, but not others ideally
                if field in ['LotArea', 'GrLivArea'] and val <= 0: # Lot/Living area must be positive
                     raise ValueError(f"{field} must be positive.")
                if field == 'TotalBsmtSF' and val < 0:
                     raise ValueError(f"{field} must be non-negative.")
                if field == 'TotalBath' and val < 0:
                     raise ValueError(f"{field} cannot be negative.")

                input_data[field] = val
            except (ValueError, TypeError, KeyError) as conversion_error:
                print(f"Error: Invalid value for field '{field}': {data.get(field)}. Details: {conversion_error}")
                raise ValueError(f"Invalid value provided for {field}.")

        # Convert to DataFrame - MUST match the column order expected by the pipeline
        # Get the feature names from the pipeline if possible, otherwise hardcode order carefully
        # Assuming the order is the same as required_fields for simplicity here
        input_df = pd.DataFrame([input_data], columns=required_fields)
        print(f"Input DataFrame for advanced prediction (raw): \n{input_df}") # DEBUG LOG

        # --- Apply Preprocessing (Log Transform Skewed Features) ---
        # This MUST match the features log-transformed during training
        input_df_processed = input_df.copy()
        skewed_features_list = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
        print(f"Applying log1p to: {skewed_features_list}") # DEBUG LOG
        for feature in skewed_features_list:
             if feature in input_df_processed.columns:
                 # Ensure positivity before log transform; values should be >= 0 based on validation
                 input_df_processed[feature] = np.log1p(input_df_processed[feature])

        print(f"Input DataFrame for advanced prediction (features processed): \n{input_df_processed}") # DEBUG LOG

        # --- Prediction ---
        # The pipeline handles imputation and scaling internally now
        predicted_price_transformed = advanced_pipeline.predict(input_df_processed)[0]
        print(f"Log-transformed prediction from pipeline: {predicted_price_transformed}") # DEBUG LOG


        # --- Reverse Log Transform (TARGET) ---
        # IMPORTANT: This assumes the target 'SalePrice' was log-transformed during training
        # Check train_model.py to confirm if y_adv = np.log1p(df_advanced[target]) was used
        predicted_price = np.expm1(predicted_price_transformed)
        predicted_price = max(0, float(predicted_price)) # Ensure non-negative float
        predicted_price_str = f"${predicted_price:,.0f}"
        print(f"Final advanced predicted price (after expm1): {predicted_price_str}") # DEBUG LOG

        return jsonify({
            'predicted_price': predicted_price_str
        })

    except ValueError as ve:
         print(f"ValueError in /predict_advanced: {ve}")
         return jsonify({'error': str(ve), 'predicted_price': 'Input Error'}), 400 # Return 400 for bad input
    except Exception as e:
        print(f"General Error in /predict_advanced: {e}")
        traceback.print_exc() # Print full traceback to server logs
        return jsonify({'error': 'An internal server error occurred.', 'predicted_price': 'Error'}), 500


# --- Main Execution Block (for local testing) ---
