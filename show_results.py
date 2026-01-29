"""Extract and display model comparison results"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load cleaned dataset
df = pd.read_csv('cleaned_real_estate_dataset.csv')
print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Neighborhoods: {df['Neighborhood'].unique()}")
print(f"Missing Sold_Price: {df['Sold_Price'].isna().sum()}")

# Features and target
FEATURES = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Condition', 'Effective_Age', 'Neighborhood_Tier']
X = df[FEATURES]
y = df['Sold_Price']

# Same split for all models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
}

print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    results.append({
        'Model': name,
        'R² Train': f"{r2_train:.4f}",
        'R² Test': f"{r2_test:.4f}",
        'Overfit Gap': f"{r2_train - r2_test:.4f}",
        'MAE ($)': f"{mae:,.0f}",
        'Pass R²>0.75': '✓' if r2_test > 0.75 else '✗'
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Best model
best = max(results, key=lambda x: float(x['R² Test']))
print(f"\n★ SELECTED: {best['Model']} (R² Test = {best['R² Test']})")
