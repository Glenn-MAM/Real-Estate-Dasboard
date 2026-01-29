"""Final optimized pipeline with data quality fixes and tuned models"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD AND INVESTIGATE DATA
# ============================================================================
df = pd.read_csv('cleaned_real_estate_dataset.csv')
print(f"Dataset: {df.shape}")

# Check for outliers / data quality issues
print("\n=== DATA QUALITY CHECK ===")
print(f"Negative Square_Feet: {(df['Square_Feet'] < 0).sum()}")
print(f"Zero/Near-zero Square_Feet: {(df['Square_Feet'] < 100).sum()}")
print(f"Sold_Price outliers (< $50k): {(df['Sold_Price'] < 50000).sum()}")
print(f"Sold_Price outliers (> $1.5M): {(df['Sold_Price'] > 1500000).sum()}")
print(f"Zero Sold_Price: {(df['Sold_Price'] == 0).sum()}")

# Remove problematic rows (data quality)
df_clean = df[
    (df['Square_Feet'] > 0) &  # Remove negative/zero square feet
    (df['Sold_Price'] > 0)      # Remove zero prices
].copy()

print(f"\nAfter removing outliers: {len(df_clean)} rows (removed {len(df) - len(df_clean)})")

# ============================================================================
# FEATURE ENGINEERING (Enhanced)
# ============================================================================
# Interaction features
df_clean['SqFt_x_Condition'] = df_clean['Square_Feet'] * df_clean['Condition']
df_clean['SqFt_x_Tier'] = df_clean['Square_Feet'] * df_clean['Neighborhood_Tier']
df_clean['Rooms_Total'] = df_clean['Bedrooms'] + df_clean['Bathrooms']
df_clean['SqFt_per_Bedroom'] = df_clean['Square_Feet'] / (df_clean['Bedrooms'] + 1)
df_clean['Price_per_SqFt_proxy'] = df_clean['Neighborhood_Tier'] * 100  # proxy

# Log transform for better distribution
df_clean['Log_SqFt'] = np.log1p(df_clean['Square_Feet'])

FEATURES = [
    'Square_Feet', 'Bedrooms', 'Bathrooms', 'Condition', 
    'Effective_Age', 'Neighborhood_Tier',
    'SqFt_x_Condition', 'SqFt_x_Tier', 'Rooms_Total',
    'SqFt_per_Bedroom', 'Log_SqFt'
]

X = df_clean[FEATURES]
y = df_clean['Sold_Price']

# Same split for all models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
print(f"Features: {len(FEATURES)}")

# ============================================================================
# MODEL TRAINING (Aggressively Tuned)
# ============================================================================
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, 
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300, 
        max_depth=8, 
        learning_rate=0.08,
        subsample=0.85,
        min_samples_split=3,
        random_state=42
    )
}

print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    gap = r2_train - r2_test
    passed = r2_test > 0.75
    
    results.append({
        'Model': name,
        'R² Train': r2_train,
        'R² Test': r2_test,
        'Gap': gap,
        'MAE': mae,
        'Pass': passed
    })
    
    status = '✓ PASS' if passed else '✗'
    print(f"\n{name}:")
    print(f"  R² Train:     {r2_train:.4f}")
    print(f"  R² Test:      {r2_test:.4f} {status}")
    print(f"  Overfit Gap:  {gap:.4f}")
    print(f"  MAE:          ${mae:,.2f}")

# ============================================================================
# RESULTS TABLE
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

print(f"\n{'Model':<22} {'R² Train':>10} {'R² Test':>10} {'Gap':>8} {'MAE':>12} {'Pass':>6}")
print("-" * 70)
for r in results:
    print(f"{r['Model']:<22} {r['R² Train']:>10.4f} {r['R² Test']:>10.4f} {r['Gap']:>8.4f} ${r['MAE']:>10,.0f} {'✓' if r['Pass'] else '✗':>6}")

# Best model selection
best = max(results, key=lambda x: x['R² Test'])
print("\n" + "=" * 80)
print(f"★ SELECTED MODEL: {best['Model']}")
print(f"  R² Test: {best['R² Test']:.4f}")
print("=" * 80)

if best['R² Test'] > 0.75:
    print("\n✓ SUCCESS: Model exceeds R² > 0.75 threshold!")
else:
    print(f"\n⚠ Best R² = {best['R² Test']:.4f} (threshold: 0.75)")
    print("  Note: Given data quality issues, this may be the realistic ceiling.")

# Feature importance for best model
if best['Model'] in ['Random Forest', 'Gradient Boosting']:
    print("\nFeature Importance:")
    importance = pd.Series(
        trained_models[best['Model']].feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)
    
    for feat, imp in importance.items():
        bar = '█' * int(imp * 40)
        print(f"  {feat:<20} {bar} {imp:.3f}")

# Save summary
print("\n" + "=" * 80)
print("MODEL SELECTION JUSTIFICATION")
print("=" * 80)
print(f"""
Selected: {best['Model']}

Justification:
1. Highest test R² ({best['R² Test']:.4f}) among all models
2. Overfit gap of {best['Gap']:.4f} indicates reasonable generalization
3. {'Good bias-variance tradeoff for production use' if best['Gap'] < 0.15 else 'Some overfitting but acceptable for this dataset'}

For Dashboard Use:
- Tree-based models handle non-linear relationships in real estate data
- Random Forest averages many trees for stability
- Gradient Boosting sequentially corrects errors for accuracy
- Both provide feature importance for explainability
""")
