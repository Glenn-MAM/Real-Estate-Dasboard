"""Simple model comparison with text output to file"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load cleaned dataset
df = pd.read_csv('cleaned_real_estate_dataset.csv')

# Remove rows with negative/zero Square_Feet or zero Sold_Price
df_clean = df[(df['Square_Feet'] > 0) & (df['Sold_Price'] > 0)].copy()

# Feature engineering
df_clean['SqFt_x_Condition'] = df_clean['Square_Feet'] * df_clean['Condition']
df_clean['SqFt_x_Tier'] = df_clean['Square_Feet'] * df_clean['Neighborhood_Tier']
df_clean['Rooms_Total'] = df_clean['Bedrooms'] + df_clean['Bathrooms']
df_clean['SqFt_per_Bedroom'] = df_clean['Square_Feet'] / (df_clean['Bedrooms'] + 1)
df_clean['Log_SqFt'] = np.log1p(df_clean['Square_Feet'])

FEATURES = [
    'Square_Feet', 'Bedrooms', 'Bathrooms', 'Condition', 
    'Effective_Age', 'Neighborhood_Tier',
    'SqFt_x_Condition', 'SqFt_x_Tier', 'Rooms_Total',
    'SqFt_per_Bedroom', 'Log_SqFt'
]

X = df_clean[FEATURES]
y = df_clean['Sold_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=2, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.08, subsample=0.85, random_state=42)
}

# Run and collect results
output_lines = []
output_lines.append("MODEL COMPARISON RESULTS")
output_lines.append("=" * 60)
output_lines.append(f"Dataset: {len(df_clean)} rows after outlier removal")
output_lines.append(f"Train: {len(X_train)}, Test: {len(X_test)}")
output_lines.append(f"Features: {len(FEATURES)}")
output_lines.append("")

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    gap = r2_train - r2_test
    
    results.append({
        'Model': name,
        'R2_Train': r2_train,
        'R2_Test': r2_test,
        'Gap': gap,
        'MAE': mae,
        'Pass': 'YES' if r2_test > 0.75 else 'NO'
    })

# Format table
output_lines.append(f"{'Model':<22} {'R2 Train':>10} {'R2 Test':>10} {'Gap':>8} {'MAE':>12} {'Pass':>6}")
output_lines.append("-" * 70)
for r in results:
    output_lines.append(f"{r['Model']:<22} {r['R2_Train']:>10.4f} {r['R2_Test']:>10.4f} {r['Gap']:>8.4f} ${r['MAE']:>10,.0f} {r['Pass']:>6}")

output_lines.append("")
output_lines.append("=" * 60)

# Best model
best = max(results, key=lambda x: x['R2_Test'])
output_lines.append(f"SELECTED MODEL: {best['Model']}")
output_lines.append(f"R2 Test Score: {best['R2_Test']:.4f}")
output_lines.append(f"Threshold (0.75): {'PASSED' if best['R2_Test'] > 0.75 else 'NOT MET'}")

# Write to file
with open('model_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

# Also print
for line in output_lines:
    print(line)
