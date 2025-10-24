import pandas as pd
import joblib
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer # Added FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # Added SimpleImputer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')
print("Successfully loaded 'train.csv'.")


print("Starting the clustering process...")
features_for_clustering = [
    'Neighborhood', 'BldgType', 'OverallQual', 'GrLivArea', 'TotalBsmtSF',
    'KitchenQual', 'YearBuilt', 'YrSold', 'MoSold', 'Street', 'SalePrice'
]
# Handle potential missing values before clustering (using median for numeric, most frequent for categorical)
df_clustering_imputed = df[features_for_clustering].copy()

numerical_features_cluster = df_clustering_imputed.select_dtypes(include=np.number).columns.tolist()
numerical_features_cluster.remove('YrSold')
numerical_features_cluster.remove('MoSold')
numerical_features_cluster.remove('SalePrice') # Keep SalePrice for potential analysis but not direct scaling/encoding here
numerical_features_cluster.remove('OverallQual') # Often treated as ordinal/categorical, depends on approach
numerical_features_cluster.remove('YearBuilt') # Often treated as ordinal/categorical, depends on approach


categorical_features_cluster = df_clustering_imputed.select_dtypes(include='object').columns.tolist()
# Add potentially ordinal features treated as categorical for one-hot encoding here if desired
categorical_features_cluster.extend(['OverallQual', 'YearBuilt', 'KitchenQual', 'Street', 'BldgType']) # Example

# Impute numerical features
num_imputer = SimpleImputer(strategy='median')
df_clustering_imputed[numerical_features_cluster] = num_imputer.fit_transform(df_clustering_imputed[numerical_features_cluster])

# Impute categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')
df_clustering_imputed[categorical_features_cluster] = cat_imputer.fit_transform(df_clustering_imputed[categorical_features_cluster])


# Redefine features for preprocessor based on imputation results
numerical_features = ['GrLivArea', 'TotalBsmtSF'] # Features to scale
categorical_features = ['Neighborhood', 'BldgType', 'KitchenQual', 'Street', 'OverallQual', 'YearBuilt'] # Features to encode


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Pass through YrSold, MoSold, SalePrice if needed by KMeans (though usually not scaled/encoded)
)

# --- KMeans Pipeline ---
k_means_model = KMeans(n_clusters=5, random_state=42, n_init='auto')
clustering_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', k_means_model)
])

try:
    # Use the imputed data for fitting the clustering pipeline
    # Select only the features needed by the preprocessor + YrSold, MoSold, SalePrice if remainder='passthrough'
    features_for_fitting = numerical_features + categorical_features + ['YrSold', 'MoSold', 'SalePrice']
    df['Cluster'] = clustering_pipeline.fit_predict(df_clustering_imputed[features_for_fitting])
    print("Clustering complete. Houses have been assigned to categories.")
except Exception as e:
    print(f"An error occurred during clustering: {e}")
    # Consider more specific error handling or logging
    # exit() # Maybe don't exit, allow rest of script to run if possible?

# --- Cluster Descriptions ---
cluster_descriptions = {}
numerical_features_to_describe = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'SalePrice']
# Ensure 'Cluster' column exists before grouping
if 'Cluster' in df.columns:
    for cluster_id, group_df in df.groupby('Cluster'):
        # Handle potential missing values in descriptive features before calculating mean
        group_df_cleaned = group_df[numerical_features_to_describe].fillna(group_df[numerical_features_to_describe].median())
        cluster_stats = group_df_cleaned.mean().to_dict()
        description = (
            f"Avg. Quality {cluster_stats['OverallQual']:.1f}, "
            f"Avg. Living Area {cluster_stats['GrLivArea']:.0f} sqft, "
            f"Avg. Price ${cluster_stats['SalePrice']:,.0f}."
        )
        cluster_descriptions[str(cluster_id)] = description
    with open('data_descriptions.json', 'w') as f:
        json.dump(cluster_descriptions, f, indent=4)
    print("Cluster descriptions saved to 'data_descriptions.json'.")
else:
    print("Warning: 'Cluster' column not found. Skipping description generation.")


# --- Simple Model Training ---
# Ensure 'Cluster' column exists before proceeding
if 'Cluster' in df.columns:
    df['Neighborhood_and_Category'] = df['Neighborhood'] + '_' + df['Cluster'].astype(str)

    avg_lot_areas = df.groupby('Neighborhood_and_Category')['LotArea'].mean().fillna(df['LotArea'].mean()).to_dict() # Fill NaN averages
    with open('avg_lot_areas.json', 'w') as f:
        json.dump(avg_lot_areas, f, indent=4)
    print("Average lot areas saved to 'avg_lot_areas.json'.")

    features_for_prediction = ['Neighborhood_and_Category', 'LotArea', 'OverallQual']
    target = 'SalePrice'

    # Drop rows where target or key features are missing for this model
    df_final = df[features_for_prediction + [target]].dropna(subset=features_for_prediction + [target]).copy()

    X = df_final.drop(target, axis=1)
    y = df_final[target]

    # Impute LotArea and OverallQual before scaling
    simple_num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor_final = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Neighborhood_and_Category']),
            ('num', simple_num_pipeline, ['LotArea', 'OverallQual']) # Apply imputation and scaling
        ],
         remainder='passthrough' # Ensure no columns are dropped unexpectedly
    )

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, objective='reg:squarederror')
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_final),
        ('regressor', xgb_model)
    ])

    print("\nTraining the SIMPLE price prediction model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        final_pipeline.fit(X_train, y_train)
        print("Simple model training complete.")

        # --- Simple Model Evaluation ---
        y_pred = final_pipeline.predict(X_test)

        # Ensure no negative predictions
        y_pred[y_pred < 0] = 0

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        # evs = explained_variance_score(y_test, y_pred) # Can be sensitive, sometimes less informative than R2

        print("\n--- Simple Model Evaluation ---")
        print(f"R-squared (R²):                  {r2:.4f}")
        # print(f"Explained Variance Score:        {evs:.4f}")
        print(f"Mean Absolute Error (MAE):       ${mae:,.2f}")
        print(f"Median Absolute Error (MedAE):   ${medae:,.2f}")
        # print(f"Mean Squared Error (MSE):        {mse:,.2f}") # Less interpretable than RMSE
        print(f"Root Mean Squared Error (RMSE):  ${rmse:,.2f}")
        print(f"Mean Absolute Percentage Error:  {mape:.2%}")
        print("----------------------------------------")


        print("\n--- Sample Predictions (Simple Model Test Set) ---")
        results_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
        results_df['Difference'] = results_df['Predicted Price'] - results_df['Actual Price']

        results_df['Actual Price'] = results_df['Actual Price'].map('${:,.0f}'.format)
        results_df['Predicted Price'] = results_df['Predicted Price'].map('${:,.0f}'.format)
        results_df['Difference'] = results_df['Difference'].map('${:,.0f}'.format)

        print(results_df.head(10).to_string(index=False))
        print("----------------------------------------------------")


        model_file_name = 'category_based_predictor.pkl'
        joblib.dump(final_pipeline, model_file_name)
        print(f"\nSimple model saved to '{model_file_name}'.")

    except Exception as e:
        print(f"An error occurred during simple model training or evaluation: {e}")

    # Save processed data *after* simple model section
    df.to_pickle('processed_data.pkl')
    print("Processed DataFrame saved to 'processed_data.pkl'.")

else:
    print("Warning: 'Cluster' column not found. Skipping Simple Model training and saving.")


# --- Advanced Model ---

print("\n" + "="*50)
print("Pre-processing, Training, and Evaluating the ADVANCED price prediction model...")

# --- 1. Engineer new features ---
# Recalculate AgeSold safely
df['AgeSold'] = df['YrSold'] - df['YearBuilt']
# Ensure AgeSold is not negative (e.g., if YrSold < YearBuilt, although unlikely)
df['AgeSold'] = df['AgeSold'].apply(lambda x: max(x, 0))

# Recalculate TotalBath, handling potential NaNs in individual components first
bath_cols = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
for col in bath_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0) # Fill NaNs with 0 before calculation

df['TotalBath'] = df.get('BsmtFullBath', 0) + \
                  (0.5 * df.get('BsmtHalfBath', 0)) + \
                  df.get('FullBath', 0) + \
                  (0.5 * df.get('HalfBath', 0))

# --- 2. Define features and target ---
features_for_advanced_model = [
    'Neighborhood',
    'LotArea',
    'OverallQual',
    'GrLivArea',
    'TotalBsmtSF', # Keep this, will impute later
    'TotalBath',
    'AgeSold'
]
target = 'SalePrice'
numerical_features_adv = ['LotArea', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'TotalBath', 'AgeSold']
categorical_features_adv = ['Neighborhood']

# Select data, handle potential missing target values early
df_advanced = df[features_for_advanced_model + [target]].copy().dropna(subset=[target])

# --- 3. More Rigorous Pre-processing ---
# Handle potential infinite values if they exist
df_advanced.replace([np.inf, -np.inf], np.nan, inplace=True)

# Address Skewness with Log Transformation (TARGET variable)
# Ensure target is positive before log transform
if (df_advanced[target] > 0).all():
    df_advanced[target] = np.log1p(df_advanced[target])
else:
    print("Warning: Target variable contains non-positive values. Log transform skipped.")
    # Consider alternative transformations or handling non-positive prices if they exist

# Separate features and target *after* target log transform
X_adv = df_advanced[features_for_advanced_model]
y_adv = df_advanced[target] # Target is potentially log-transformed

# Address Skewness in FEATURES (before imputation/scaling)
X_adv_processed = X_adv.copy()
skewed_features_list = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
for feature in skewed_features_list:
    if feature in X_adv_processed.columns:
        # Impute with 0 temporarily ONLY for the positivity check before log transform
        temp_col_check = X_adv_processed[feature].fillna(0)
        if (temp_col_check >= 0).all():
             # Apply log1p using apply to handle potential NaNs correctly if needed, though imputation happens next
             X_adv_processed[feature] = X_adv_processed[feature].apply(lambda x: np.log1p(x) if pd.notna(x) and x >= 0 else x)
        else:
             print(f"Warning: Feature '{feature}' contains negative values. Log transform skipped.")

# --- 4. Define Preprocessor with Imputation & Scaling ---
# Pipeline for numerical features: Impute -> Scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Handle missing numeric values
    ('scaler', StandardScaler())
])

# Pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing categories
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use dense output for XGBoost
])

# Create the main preprocessor
preprocessor_advanced = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_adv),
        ('cat', categorical_transformer, categorical_features_adv)
    ],
    remainder='passthrough' # Check if any columns are unintentionally passed
)

# --- 5. Create the XGBoost Regressor and Final Pipeline ---
xgb_model_advanced = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, objective='reg:squarederror')

advanced_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_advanced),
    ('regressor', xgb_model_advanced)
])

# --- 6. Split Data for Evaluation (using processed features) ---
# Use X_adv_processed which has log transforms applied to features
X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(
    X_adv_processed, y_adv, test_size=0.2, random_state=42
)

# --- 7. Train on Training Set ---
print("Training advanced model on the training split...")
try:
    advanced_pipeline.fit(X_train_adv, y_train_adv) # y_train_adv is potentially log-transformed
    print("Advanced model training complete.")

    # --- 8. Evaluate on Test Set ---
    print("\nEvaluating advanced model on the test split...")
    y_pred_transformed = advanced_pipeline.predict(X_test_adv)

    # Reverse the log transformation if it was applied to the target
    y_pred_adv = np.expm1(y_pred_transformed) if (df_advanced[target] > 0).all() else y_pred_transformed
    y_test_adv_actual = np.expm1(y_test_adv) if (df_advanced[target] > 0).all() else y_test_adv

    # Handle potential negative predictions after expm1
    y_pred_adv[y_pred_adv < 0] = 0

    # Calculate Metrics
    adv_mse = mean_squared_error(y_test_adv_actual, y_pred_adv)
    adv_rmse = np.sqrt(adv_mse)
    adv_mae = mean_absolute_error(y_test_adv_actual, y_pred_adv)
    adv_r2 = r2_score(y_test_adv_actual, y_pred_adv)
    # MAPE can be problematic if actual values are close to zero
    # Filter out near-zero actual values for MAPE calculation
    mask = y_test_adv_actual > 1 # Avoid division by zero or near-zero
    if mask.sum() > 0:
        adv_mape = mean_absolute_percentage_error(y_test_adv_actual[mask], y_pred_adv[mask])
    else:
        adv_mape = np.nan # Not calculable

    print("\n--- Advanced Model Evaluation ---")
    print(f"R-squared (R²):                  {adv_r2:.4f}")
    print(f"Mean Absolute Error (MAE):       ${adv_mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE):  ${adv_rmse:,.2f}")
    if not np.isnan(adv_mape):
        print(f"Mean Absolute Percentage Error:  {adv_mape:.2%}")
    else:
        print("Mean Absolute Percentage Error:  N/A (due to zero actual values)")
    print("---------------------------------")
    if (df_advanced[target] > 0).all():
        print("(Metrics calculated on actual dollar values after reversing log transform)")
    else:
         print("(Metrics calculated on potentially non-log-transformed values)")


    # --- 9. Retrain on Full Data & Save ---
    print("\nRetraining advanced model on ALL pre-processed data for deployment...")
    # Fit the pipeline on the entire pre-processed dataset
    advanced_pipeline.fit(X_adv_processed, y_adv) # Use processed features and potentially log-transformed target

    advanced_model_file_name = 'advanced_predictor.pkl'
    joblib.dump(advanced_pipeline, advanced_model_file_name)
    print(f"Final advanced model saved to '{advanced_model_file_name}'.")

except Exception as e:
    print(f"An error occurred during advanced model training or evaluation: {e}")

print("="*50 + "\n")

# ^^^^^^^^^^^^ END OF REPLACEMENT CODE ^^^^^^^^^^^^


# --- Plotting for SIMPLE model ---
# Check if simple model evaluation was successful before plotting
if 'y_test' in locals() and 'y_pred' in locals():
    print("\nGenerating model performance plots (for the SIMPLE model)...")
    try:
        plt.figure(figsize=(18, 6))
        # Use a currently available style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            print("Seaborn style not found, using default.")
            plt.style.use('default')

        # Plot 1: Actual vs. Predicted Prices (Test Set) - Simple Model
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='royalblue', s=50, edgecolor='w') # Added s and edgecolor
        # Add limits for better visualization
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
        plt.xlim(min_val * 0.9, max_val * 1.1) # Add some padding
        plt.ylim(min_val * 0.9, max_val * 1.1)
        plt.title('Simple Model: Actual vs. Predicted Prices (Test Set)', fontsize=14, fontweight='bold') # Added bold
        plt.xlabel('Actual Sale Price ($)', fontsize=12)
        plt.ylabel('Predicted Sale Price ($)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6) # Customize grid


        # Plot 2: Distribution of Prediction Errors (Residuals) - Simple Model
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True, color='forestgreen', bins=30) # Specified bins
        plt.axvline(0, color='red', linestyle='--', lw=2)
        plt.title('Simple Model: Distribution of Prediction Errors (Test Set)', fontsize=14, fontweight='bold') # Added bold
        plt.xlabel('Prediction Error (Residuals) ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6) # Customize grid

        plt.tight_layout(pad=3.0) # Add padding between plots
        plt.show()
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
else:
    print("\nSkipping plotting due to issues in simple model evaluation.")


# --- LIMITATIONS DISCUSSION ---
# ==============================
"""
While this project provides useful house price predictions, several limitations
should be considered:

1.  Data Scope & Time:
    * The model is trained solely on the provided 'train.csv' dataset. Market conditions change; accuracy will likely degrade over time without retraining on newer data.
    * May not generalize well to different housing markets or property types.

2.  Feature Set:
    * Relies on a limited set of input features. Factors like specific condition, school quality, crime rates, amenities, interior details are not included.
    * Simple Model's 'Cluster' is an abstraction; houses within categories vary.

3.  Clustering Assumptions:
    * KMeans assumes spherical clusters. The choice of 5 clusters was pre-determined.

4.  Model Simplifications:
    * Simple Model uses average lot size when none is provided (an approximation).
    * Advanced Model's log transformation assumes a log-normal relationship for price/features, which might not hold perfectly. Imputation introduces estimated values.

5.  External Factors:
    * Doesn't account for macroeconomic factors (interest rates, economy) or local changes (zoning).

6.  Evaluation Metrics:
    * RMSE is sensitive to outliers. MAE provides a view of typical error magnitude. MAPE can be unstable if actual prices are near zero. R-squared indicates proportion of variance explained.
"""
# ==============================

print("\n--- Project Limitations ---")
print("See the comments at the end of 'train_model.py' for a discussion on limitations.")
print("--------------------------")