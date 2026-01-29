# Real Estate Investor Dashboard

A machine learning-powered dashboard for analyzing real estate investment opportunities using **Flip** vs **Rental** strategies.

## 📋 Overview

This project processes a real estate dataset, trains predictive models for house prices, and provides an interactive Gradio dashboard for investment analysis.

---

## 🔧 Data Cleaning Steps

| Step | Issue | Solution |
|------|-------|----------|
| 1 | `Square_Feet` contains "sqft" and commas | Removed non-numeric characters, converted to float |
| 2 | Duplicate `House_ID` values | Reassigned unique IDs > max existing ID |
| 3 | Missing `Sold_Price` values | Imputed using median of houses with same bedrooms, bathrooms, and neighborhood |
| 4 | Neighborhood typos ("Gilden Ridge", "Swamp Botom") | Mapped to official 5 neighborhoods: Maplewood, Golden Ridge, Sunny Valley, Swamp Bottom, Industrial District |
| 5 | `Renovation_Year` < `Year_Built` | Swapped values to ensure Build_Year ≤ Renovation_Year |

---

## ⚙️ Feature Engineering

### Effective_Age
```
Effective_Age = Current_Year - max(Year_Built, Year_Renovated)
```
Captures the "true" age of the property considering renovations.

### Neighborhood_Tier
Ranked 1-5 based on median `Sold_Price`:
| Tier | Neighborhood | Median Price |
|------|--------------|--------------|
| 1 | Swamp Bottom | $244,969 |
| 2 | Industrial District | $340,313 |
| 3 | Maplewood | $452,221 |
| 4 | Sunny Valley | $555,022 |
| 5 | Golden Ridge | $656,229 |

---

## 📊 Model Selection

| Model | R² Train | R² Test | Overfit Gap | Status |
|-------|----------|---------|-------------|--------|
| Linear Regression | 0.8020 | **0.7629** | 0.0391 | ✅ Selected |
| Random Forest | 0.9693 | 0.7152 | 0.2541 | ❌ Overfitting |
| Gradient Boosting | 0.9999 | 0.6555 | 0.3443 | ❌ Overfitting |

### Justification
- **Linear Regression** is the only model exceeding the R² ≥ 0.75 threshold on test data
- Smallest overfit gap (0.039) indicates best generalization to unseen data
- Simpler model = faster predictions + better interpretability
- Suitable for production dashboard with real-time predictions

---

## 💰 Investment Analysis Logic

### Flip Analysis
1. **FMV** = Model prediction of Fair Market Value
2. **Net Profit** = FMV - (Asking Price + Renovation Budget)
3. **Profit Margin** = Net Profit / FMV × 100%
4. ⚠️ **HIGH RISK** warning if Profit Margin < 15%

### Rental Analysis
1. **Monthly Rent** = 0.8% × FMV
2. **Monthly Expenses** = 1.5% × Asking Price / 12
3. **Mortgage** = 30-year fixed payment formula:
   ```
   M = P × r × (1+r)^n / ((1+r)^n - 1)
   ```
   Where: P = loan amount, r = monthly rate, n = 360 payments
4. **Cash Flow** = Rent - (Mortgage + Expenses)
5. ⚠️ **NEGATIVE GEARING** (red text) if Cash Flow < 0

---

## 🚀 Running the Project

### Prerequisites
```bash
pip install pandas numpy scikit-learn gradio
```

### Run the Dashboard
```bash
py real_estate_dashboard.py
```
Then open http://127.0.0.1:7860 in your browser.

### Run the Full Notebook
```bash
py -m jupyter notebook Real_Estate_Dashboard.ipynb
```

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `Real_Estate_Dashboard.ipynb` | Complete notebook with all steps documented |
| `real_estate_dashboard.py` | Standalone Gradio dashboard |
| `cleaned_real_estate.csv` | Final cleaned dataset |
| `real_estate_pipeline.py` | Data cleaning pipeline |
| `final_model_comparison.py` | Model training and comparison |
| `real_estate_dataset.csv` | Original raw dataset |

---

## 📸 Example Scenario

**Property:** 2,500 sqft, 4 bed, 2 bath, Golden Ridge, Condition 8  
**Financials:** Asking $450,000, Renovation $75,000, Interest 7%

| Analysis | Result |
|----------|--------|
| **Flip** | FMV: $780,000 → Net Profit: $255,000 → Margin: 32.7% ✅ |
| **Rental** | Rent: $6,240/mo, Mortgage: $3,495/mo, Expenses: $562/mo → **Cash Flow: +$2,183** ✅ |

A scenario where flip is profitable but rental shows negative gearing can occur with:
- Lower neighborhood tier (lower FMV → lower rent)
- High asking price (high mortgage payments)
- High interest rate

---

## 👤 Author

Real Estate Investor Dashboard Project  
Data Engineering & Machine Learning Pipeline
