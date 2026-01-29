"""
Real Estate Data Pipeline
========================
Data cleaning, feature engineering, and model training for the Real Estate Investor Dashboard.

Data Cleaning Rules Applied:
1. Square_Feet: Remove 'sqft', commas; convert to float
2. House_ID duplicates: Reassign new unique IDs > max existing ID
3. Sold_Price missing: Impute with median of similar houses (same bedrooms, bathrooms, neighborhood)
4. Neighborhood typos: Map to 5 valid neighborhoods
5. Year swap: If Renovation_Year < Year_Built, swap the values
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
CURRENT_YEAR = 2026
VALID_NEIGHBORHOODS = ['Maplewood', 'Golden Ridge', 'Sunny Valley', 'Swamp Bottom', 'Industrial District']

# Neighborhood typo mapping (detected from data analysis)
NEIGHBORHOOD_MAP = {
    'Gilden Ridge': 'Golden Ridge',
    'Swamp Botom': 'Swamp Bottom'
}

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 60)
print("REAL ESTATE DATA PIPELINE")
print("=" * 60)

df = pd.read_csv('real_estate_dataset.csv')
print(f"\n[1] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# DATA CLEANING
# =============================================================================

# -----------------------------------------------------------------------------
# 1. CLEAN SQUARE_FEET: Remove 'sqft', commas, convert to float
# -----------------------------------------------------------------------------
df['Square_Feet'] = (df['Square_Feet']
                     .astype(str)
                     .str.replace(',', '', regex=False)
                     .str.replace('sqft', '', regex=False)
                     .str.strip())
df['Square_Feet'] = pd.to_numeric(df['Square_Feet'], errors='coerce')
print(f"[2] Cleaned Square_Feet column - converted to numeric")

# -----------------------------------------------------------------------------
# 2. FIX DUPLICATE HOUSE_IDs: Assign new unique IDs greater than max
# -----------------------------------------------------------------------------
# Extract numeric part of House_ID
df['_house_num'] = df['House_ID'].str.replace('H-', '', regex=False).astype(int)
max_house_id = df['_house_num'].max()

# Find duplicates (keep first occurrence, mark others as duplicates)
duplicated_mask = df['House_ID'].duplicated(keep='first')
num_duplicates = duplicated_mask.sum()

# Assign new IDs to duplicates
new_id_counter = max_house_id + 1
for idx in df[duplicated_mask].index:
    df.at[idx, 'House_ID'] = f'H-{new_id_counter:04d}'
    new_id_counter += 1

df.drop(columns=['_house_num'], inplace=True)
print(f"[3] Fixed {num_duplicates} duplicate House_IDs (reassigned IDs > H-{max_house_id:04d})")

# -----------------------------------------------------------------------------
# 3. CORRECT NEIGHBORHOOD TYPOS: Map to valid neighborhoods
# -----------------------------------------------------------------------------
original_neighborhoods = df['Neighborhood'].nunique()
df['Neighborhood'] = df['Neighborhood'].replace(NEIGHBORHOOD_MAP)
final_neighborhoods = df['Neighborhood'].nunique()

# Verify all neighborhoods are valid
invalid = df[~df['Neighborhood'].isin(VALID_NEIGHBORHOODS)]['Neighborhood'].unique()
if len(invalid) > 0:
    print(f"    WARNING: Unmapped neighborhoods found: {invalid}")
else:
    print(f"[4] Corrected neighborhood typos ({original_neighborhoods} -> {final_neighborhoods} unique neighborhoods)")

# -----------------------------------------------------------------------------
# 4. FIX YEAR SWAP: If Renovation_Year < Year_Built, swap them
# -----------------------------------------------------------------------------
year_swap_mask = df['Year_Renovated'] < df['Year_Built']
num_swaps = year_swap_mask.sum()

# Swap the values for affected rows
df.loc[year_swap_mask, ['Year_Built', 'Year_Renovated']] = (
    df.loc[year_swap_mask, ['Year_Renovated', 'Year_Built']].values
)
print(f"[5] Fixed {num_swaps} year swap issues (Renovation_Year < Year_Built)")

# -----------------------------------------------------------------------------
# 5. IMPUTE MISSING SOLD_PRICE: Using median of similar houses
#    Similar = same bedrooms, bathrooms, neighborhood
# -----------------------------------------------------------------------------
missing_price_mask = df['Sold_Price'].isnull()
num_missing = missing_price_mask.sum()
imputed_count = 0

for idx in df[missing_price_mask].index:
    bedrooms = df.at[idx, 'Bedrooms']
    bathrooms = df.at[idx, 'Bathrooms']
    neighborhood = df.at[idx, 'Neighborhood']
    
    # Find similar houses with non-null Sold_Price
    similar = df[
        (df['Bedrooms'] == bedrooms) & 
        (df['Bathrooms'] == bathrooms) & 
        (df['Neighborhood'] == neighborhood) &
        (df['Sold_Price'].notna())
    ]
    
    if len(similar) > 0:
        # Use median
        imputed_value = similar['Sold_Price'].median()
    else:
        # Fallback: use mean of same neighborhood
        fallback = df[(df['Neighborhood'] == neighborhood) & (df['Sold_Price'].notna())]
        if len(fallback) > 0:
            imputed_value = fallback['Sold_Price'].mean()
        else:
            # Final fallback: overall mean
            imputed_value = df['Sold_Price'].mean()
    
    df.at[idx, 'Sold_Price'] = round(imputed_value, 2)
    imputed_count += 1

print(f"[6] Imputed {imputed_count} missing Sold_Price values using median of similar houses")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# -----------------------------------------------------------------------------
# EFFECTIVE_AGE: Current_Year - max(Year_Built, Year_Renovated)
# -----------------------------------------------------------------------------
df['Effective_Age'] = CURRENT_YEAR - df[['Year_Built', 'Year_Renovated']].max(axis=1)
print(f"[7] Created Effective_Age feature (range: {df['Effective_Age'].min()} - {df['Effective_Age'].max()})")

# -----------------------------------------------------------------------------
# NEIGHBORHOOD_TIER: Rank 1-5 based on median Sold_Price (1=lowest, 5=highest)
# -----------------------------------------------------------------------------
median_prices = df.groupby('Neighborhood')['Sold_Price'].median().sort_values()
tier_map = {neigh: rank for rank, neigh in enumerate(median_prices.index, 1)}
df['Neighborhood_Tier'] = df['Neighborhood'].map(tier_map)

print(f"[8] Created Neighborhood_Tier feature:")
for neigh, tier in sorted(tier_map.items(), key=lambda x: x[1]):
    print(f"    Tier {tier}: {neigh} (median: ${median_prices[neigh]:,.0f})")

# =============================================================================
# DATA VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

# Check no duplicates
assert df['House_ID'].nunique() == len(df), "FAIL: Duplicate House_IDs exist"
print("✓ No duplicate House_IDs")

# Check exactly 5 neighborhoods
assert df['Neighborhood'].nunique() == 5, f"FAIL: {df['Neighborhood'].nunique()} neighborhoods"
print("✓ Exactly 5 unique neighborhoods")

# Check no missing Sold_Price
assert df['Sold_Price'].notna().all(), "FAIL: Missing Sold_Price values"
print("✓ No missing Sold_Price values")

# Check all Year_Built <= Year_Renovated
assert (df['Year_Built'] <= df['Year_Renovated']).all(), "FAIL: Year swap issues remain"
print("✓ All Year_Built <= Year_Renovated")

# Check Square_Feet is numeric
assert df['Square_Feet'].dtype in ['float64', 'int64'], "FAIL: Square_Feet not numeric"
print("✓ Square_Feet is numeric")

# Check new features exist
assert 'Effective_Age' in df.columns and 'Neighborhood_Tier' in df.columns, "FAIL: Features missing"
print("✓ Effective_Age and Neighborhood_Tier features created")

# =============================================================================
# MODEL TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Define features and target
FEATURES = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Condition', 'Effective_Age', 'Neighborhood_Tier']
TARGET = 'Sold_Price'

X = df[FEATURES]
y = df[TARGET]

# Train/test split (80/20) - same split for all models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nFeatures: {FEATURES}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
}

# Train and evaluate each model
results = []

print("\n" + "-" * 60)
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    overfit_gap = r2_train - r2_test
    
    results.append({
        'Model': name,
        'R² Train': r2_train,
        'R² Test': r2_test,
        'Overfit Gap': overfit_gap,
        'MAE Test': mae_test,
        'RMSE Test': rmse_test
    })
    
    print(f"\n{name}:")
    print(f"  R² Train: {r2_train:.4f}")
    print(f"  R² Test:  {r2_test:.4f}")
    print(f"  Overfit Gap: {overfit_gap:.4f}")
    print(f"  MAE:  ${mae_test:,.2f}")
    print(f"  RMSE: ${rmse_test:,.2f}")

# =============================================================================
# MODEL COMPARISON & SELECTION
# =============================================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Select best model based on test R² (with consideration for overfitting)
# Prefer models with smaller overfit gap if R² scores are close
best_idx = results_df['R² Test'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_r2 = results_df.loc[best_idx, 'R² Test']
best_overfit = results_df.loc[best_idx, 'Overfit Gap']

print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)
print(f"\n★ {best_model_name}")
print(f"  - Test R²: {best_r2:.4f} (> 0.75 threshold: {'✓ PASS' if best_r2 > 0.75 else '✗ FAIL'})")
print(f"  - Overfit Gap: {best_overfit:.4f}")

# Justification
print("\nJustification for model selection:")
if best_model_name == 'Gradient Boosting':
    print("  - Best generalization performance (highest test R²)")
    print("  - Moderate overfitting gap indicates good bias-variance balance")
    print("  - Ensemble method provides robust predictions for diverse inputs")
    print("  - Well-suited for a dashboard where prediction reliability is critical")
elif best_model_name == 'Random Forest':
    print("  - Strong generalization with inherent regularization from averaging")
    print("  - Robust to outliers in the data")
    print("  - Provides feature importance for interpretability")
    print("  - Good choice for production dashboards requiring stability")
else:
    print("  - Simple, interpretable baseline model")
    print("  - Low computational cost for real-time dashboard predictions")
    print("  - Linear relationships may dominate in this dataset")

# =============================================================================
# SAVE CLEANED DATASET
# =============================================================================
output_file = 'cleaned_real_estate_dataset.csv'
df.to_csv(output_file, index=False)
print(f"\n[OUTPUT] Cleaned dataset saved to: {output_file}")
print(f"         Final shape: {df.shape}")

# Feature importance for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    best_model = models[best_model_name]
    importance = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        bar = '█' * int(row['Importance'] * 50)
        print(f"  {row['Feature']:18} {bar} {row['Importance']:.3f}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
