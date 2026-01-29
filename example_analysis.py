"""
Example Analysis Script - Demonstrates Flip Positive / Rental Negative Scenario
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv('cleaned_real_estate_dataset.csv')
df_clean = df[(df['Square_Feet'] > 0) & (df['Sold_Price'] > 0)].copy()

FEATURES = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Condition', 'Effective_Age', 'Neighborhood_Tier']
X = df_clean[FEATURES]
y = df_clean['Sold_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
r2 = model.score(X_test, y_test)

NEIGHBORHOOD_TIER_MAP = {
    'Swamp Bottom': 1, 'Industrial District': 2, 'Maplewood': 3, 
    'Sunny Valley': 4, 'Golden Ridge': 5
}

def predict_fmv(sqft, bed, bath, neighborhood, condition):
    tier = NEIGHBORHOOD_TIER_MAP[neighborhood]
    features = np.array([[sqft, bed, bath, condition, 5, tier]])
    return max(model.predict(features)[0], 0)

def calculate_mortgage(principal, rate, years=30):
    if rate <= 0: return principal / (years * 12)
    r = rate / 100 / 12
    n = years * 12
    return principal * r * ((1+r)**n) / (((1+r)**n) - 1)

# ============================================================================
# EXAMPLE: Flip Positive, Rental Negative
# ============================================================================
print("=" * 70)
print("EXAMPLE: FLIP POSITIVE / RENTAL NEGATIVE SCENARIO")
print("=" * 70)

# Property details
sqft, bed, bath = 1800, 3, 2
neighborhood = 'Swamp Bottom'  # Low tier
condition = 6
asking_price = 250000
renovation = 30000
interest_rate = 7.5

print(f"\nPROPERTY DETAILS:")
print(f"  {sqft} sqft | {bed} bed | {bath} bath")
print(f"  Neighborhood: {neighborhood} (Tier {NEIGHBORHOOD_TIER_MAP[neighborhood]})")
print(f"  Condition: {condition}/10")

print(f"\nFINANCIALS:")
print(f"  Asking Price:      ${asking_price:,}")
print(f"  Renovation Budget: ${renovation:,}")
print(f"  Interest Rate:     {interest_rate}%")

# Calculate FMV
fmv = predict_fmv(sqft, bed, bath, neighborhood, condition)

print(f"\n" + "-" * 70)
print("FLIP ANALYSIS")
print("-" * 70)
total_investment = asking_price + renovation
net_profit = fmv - total_investment
profit_margin = (net_profit / fmv) * 100

print(f"  Fair Market Value (FMV): ${fmv:,.2f}")
print(f"  Total Investment:        ${total_investment:,}")
print(f"  Net Profit:              ${net_profit:,.2f}")
print(f"  Profit Margin:           {profit_margin:.1f}%")
if profit_margin >= 15:
    print(f"  Assessment:              [OK] ACCEPTABLE (margin >= 15%)")
else:
    print(f"  Assessment:              [!!!] HIGH RISK (margin < 15%)")

print(f"\n" + "-" * 70)
print("RENTAL ANALYSIS")
print("-" * 70)
monthly_rent = 0.008 * fmv
monthly_expenses = (0.015 * asking_price) / 12
monthly_mortgage = calculate_mortgage(asking_price + renovation, interest_rate)
cash_flow = monthly_rent - (monthly_mortgage + monthly_expenses)

print(f"  Monthly Rent:     ${monthly_rent:,.2f} (0.8% of FMV)")
print(f"  Monthly Mortgage: ${monthly_mortgage:,.2f} (30-yr @ {interest_rate}%)")
print(f"  Monthly Expenses: ${monthly_expenses:,.2f} (1.5% of asking/12)")
print(f"  ------------------------------------------")
print(f"  Cash Flow:        ${cash_flow:,.2f} / month")
if cash_flow < 0:
    print(f"  Assessment:       [!!!] NEGATIVE GEARING")
else:
    print(f"  Assessment:       [OK] POSITIVE CASH FLOW")

print(f"\n" + "=" * 70)
print("EXPLANATION")
print("=" * 70)
print("""
WHY FLIP WORKS BUT RENTAL DOESN'T:

1. FLIP SUCCESS:
   - FMV exceeds total investment by a good margin
   - Quick turnaround captures value without ongoing costs
   - Renovation adds value that offsets the purchase price

2. RENTAL FAILURE:
   - Low neighborhood tier = lower FMV = lower rent potential
   - High interest rate (7.5%) creates expensive mortgage
   - Monthly costs exceed rental income from this property type
   
RECOMMENDATION:
  → FLIP this property, do NOT rent it
  → For rental: need higher-tier neighborhood or lower purchase price
""")

print("=" * 70)
print(f"MODEL INFO: Linear Regression | R² = {r2:.4f}")
print("=" * 70)
