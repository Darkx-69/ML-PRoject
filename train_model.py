import pandas as pd
import joblib
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

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
df_clustering = df[features_for_clustering].copy()

numerical_features = df_clustering.select_dtypes(include=['int64', 'float64']).columns.tolist()

numerical_features.remove('YrSold')
numerical_features.remove('MoSold')
numerical_features.remove('SalePrice')
categorical_features = df_clustering.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
k_means_model = KMeans(n_clusters=5, random_state=42, n_init='auto')
clustering_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', k_means_model)
])

try:
    df['Cluster'] = clustering_pipeline.fit_predict(df_clustering)
    print("Clustering complete. Houses have been assigned to categories.")
except Exception as e:
    print(f"An error occurred during clustering: {e}")
    exit()

cluster_descriptions = {}
numerical_features_to_describe = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'SalePrice']
for cluster_id, group_df in df.groupby('Cluster'):
    cluster_stats = group_df[numerical_features_to_describe].mean().to_dict()
    description = (
        f"This category represents homes with an average overall quality of "
        f"{cluster_stats['OverallQual']:.1f}, an average living area of "
        f"{cluster_stats['GrLivArea']:.0f} sq. ft., and an average sale price of "
        f"${cluster_stats['SalePrice']:.0f}."
    )
    cluster_descriptions[str(cluster_id)] = description
with open('data_descriptions.json', 'w') as f:
    json.dump(cluster_descriptions, f, indent=4)
print("Cluster descriptions saved to 'data_descriptions.json'.")


df['Neighborhood_and_Category'] = df['Neighborhood'] + '_' + df['Cluster'].astype(str)

avg_lot_areas = df.groupby('Neighborhood_and_Category')['LotArea'].mean().to_dict()
with open('avg_lot_areas.json', 'w') as f:
    json.dump(avg_lot_areas, f, indent=4)
print("Average lot areas saved to 'avg_lot_areas.json'.")

features_for_prediction = ['Neighborhood_and_Category', 'LotArea', 'OverallQual']
target = 'SalePrice'
df_final = df[features_for_prediction + [target]].copy()

X = df_final.drop(target, axis=1)
y = df_final[target]
preprocessor_final = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Neighborhood_and_Category']),
        ('num', StandardScaler(), ['LotArea', 'OverallQual'])
    ]
)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_final),
    ('regressor', xgb_model)
])
print("Training the final price prediction model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_pipeline.fit(X_train, y_train)
print("Model training complete.")

y_pred = final_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\n--- Comprehensive Model Evaluation ---")
print(f"R-squared (R²):                  {r2:.4f}")
print(f"Explained Variance Score:        {evs:.4f}")
print(f"Mean Absolute Error (MAE):       ${mae:,.2f}")
print(f"Median Absolute Error (MedAE):   ${medae:,.2f}")
print(f"Mean Squared Error (MSE):        {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE):  ${rmse:,.2f}")
print(f"Mean Absolute Percentage Error:  {mape:.2%}")
print("----------------------------------------")


print("\n--- Table of Early Outcomes (Sample from Test Set) ---")
results_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
results_df['Difference'] = results_df['Predicted Price'] - results_df['Actual Price']


results_df['Actual Price'] = results_df['Actual Price'].map('${:,.2f}'.format)
results_df['Predicted Price'] = results_df['Predicted Price'].map('${:,.2f}'.format)
results_df['Difference'] = results_df['Difference'].map('${:,.2f}'.format)

print(results_df.head(10).to_string())
print("----------------------------------------------------")


model_file_name = 'category_based_predictor.pkl'
joblib.dump(final_pipeline, model_file_name)
print(f"\nModel saved to '{model_file_name}'.")

df.to_pickle('processed_data.pkl')
print("Processed DataFrame saved to 'processed_data.pkl'.")


print("\nGenerating model performance plots...")

# Get predictions for the training set as well
y_train_pred = final_pipeline.predict(X_train)

# Create a figure for the plots
plt.figure(figsize=(18, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Actual vs. Predicted Prices (Test Set)
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='royalblue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.title('Actual vs. Predicted Prices (Test Set)', fontsize=14)
plt.xlabel('Actual Sale Price ($)', fontsize=12)
plt.ylabel('Predicted Sale Price ($)', fontsize=12)

# Plot 2: Distribution of Prediction Errors (Residuals)
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='forestgreen')
plt.axvline(0, color='red', linestyle='--', lw=2)
plt.title('Distribution of Prediction Errors on Test Set', fontsize=14)
plt.xlabel('Prediction Error (Residuals) ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plots
plt.tight_layout()
plt.show()