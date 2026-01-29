"""Improve model to achieve R² > 0.75"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load cleaned dataset
df = pd.read_csv('cleaned_real_estate_dataset.csv')
print(f"Dataset: {df.shape[0]} rows")

# Enhanced feature engineering
# Add polynomial/interaction features
df['SqFt_x_Condition'] = df['Square_Feet'] * df['Condition']
df['SqFt_x_Tier'] = df['Square_Feet'] * df['Neighborhood_Tier']
df['Rooms_Total'] = df['Bedrooms'] + df['Bathrooms']
df['Age_x_Condition'] = df['Effective_Age'] * df['Condition']
df['SqFt_per_Room'] = df['Square_Feet'] / (df['Rooms_Total'] + 1)
df['SqFt_Squared'] = df['Square_Feet'] ** 2

# Features - expanded set
FEATURES = [
    'Square_Feet', 'Bedrooms', 'Bathrooms', 'Condition', 
    'Effective_Age', 'Neighborhood_Tier',
    'SqFt_x_Condition', 'SqFt_x_Tier', 'Rooms_Total',
    'Age_x_Condition', 'SqFt_per_Room', 'SqFt_Squared'
]

X = df[FEATURES]
y = df['Sold_Price']

# Same split for all models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Features: {len(FEATURES)}")

# Models with tuned hyperparameters
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=200, 
        max_depth=20, 
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
}

print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE (with enhanced features)")
print("=" * 80)

results = []
best_model = None
best_r2 = 0

for name, model in models.items():
    model.fit(X_train, y_train)
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
    
    if r2_test > best_r2:
        best_r2 = r2_test
        best_model = name
    
    status = '✓ PASS' if passed else '✗ FAIL'
    print(f"\n{name}:")
    print(f"  R² Train:     {r2_train:.4f}")
    print(f"  R² Test:      {r2_test:.4f} {status}")
    print(f"  Overfit Gap:  {gap:.4f}")
    print(f"  MAE:          ${mae:,.2f}")

print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

df_results = pd.DataFrame(results)
df_results['R² Train'] = df_results['R² Train'].apply(lambda x: f"{x:.4f}")
df_results['R² Test'] = df_results['R² Test'].apply(lambda x: f"{x:.4f}")
df_results['Gap'] = df_results['Gap'].apply(lambda x: f"{x:.4f}")
df_results['MAE'] = df_results['MAE'].apply(lambda x: f"${x:,.0f}")
df_results['Pass'] = df_results['Pass'].apply(lambda x: '✓' if x else '✗')
print(df_results.to_string(index=False))

print("\n" + "=" * 80)
print(f"★ SELECTED MODEL: {best_model}")
print(f"  R² Test: {best_r2:.4f}")
print("=" * 80)

if best_r2 > 0.75:
    print("\n✓ SUCCESS: Model exceeds R² > 0.75 threshold!")
else:
    print(f"\n✗ THRESHOLD NOT MET: Best R² = {best_r2:.4f}")
    print("  Consider: more feature engineering or hyperparameter tuning")
