"""
Real Estate Investor Dashboard
==============================
Gradio-based dashboard for Flip vs Rental investment analysis.
Uses trained Linear Regression model to predict Fair Market Value (FMV).

Business Logic:
- Flip Analysis: Calculate profit margin and flag HIGH RISK if < 15%
- Rental Analysis: Calculate cash flow and flag NEGATIVE GEARING if < 0
"""

import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA AND TRAIN MODEL
# =============================================================================

# Load the cleaned dataset
df = pd.read_csv('cleaned_real_estate_dataset.csv')

# Remove outliers (same as in model training)
df_clean = df[
    (df['Square_Feet'] > 0) & 
    (df['Sold_Price'] > 0)
].copy()

# Features used in training
FEATURES = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Condition', 'Effective_Age', 'Neighborhood_Tier']

# Neighborhood tier mapping (from data)
NEIGHBORHOOD_TIER_MAP = {
    'Swamp Bottom': 1,
    'Industrial District': 2,
    'Maplewood': 3,
    'Sunny Valley': 4,
    'Golden Ridge': 5
}

NEIGHBORHOODS = list(NEIGHBORHOOD_TIER_MAP.keys())

# Train the model (Linear Regression - best performer)
X = df_clean[FEATURES]
y = df_clean['Sold_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model trained. R² on test set: {model.score(X_test, y_test):.4f}")

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_fmv(square_feet, bedrooms, bathrooms, neighborhood, condition):
    """Predict Fair Market Value using the trained model."""
    # Get neighborhood tier
    tier = NEIGHBORHOOD_TIER_MAP.get(neighborhood, 3)
    
    # Calculate effective age (assume average renovation age for new properties)
    # For dashboard, we assume new/recently renovated property
    effective_age = 5  # Default to 5 years effective age
    
    # Create feature array
    features = np.array([[square_feet, bedrooms, bathrooms, condition, effective_age, tier]])
    
    # Predict
    fmv = model.predict(features)[0]
    return max(fmv, 0)  # Ensure non-negative

# =============================================================================
# MORTGAGE CALCULATION
# =============================================================================

def calculate_monthly_mortgage(principal, annual_rate, years=30):
    """
    Calculate monthly mortgage payment using standard amortization formula.
    P * r * (1+r)^n / ((1+r)^n - 1)
    """
    if annual_rate <= 0:
        return principal / (years * 12)
    
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    
    numerator = principal * monthly_rate * ((1 + monthly_rate) ** num_payments)
    denominator = ((1 + monthly_rate) ** num_payments) - 1
    
    return numerator / denominator

# =============================================================================
# FLIP ANALYSIS
# =============================================================================

def flip_analysis(square_feet, bedrooms, bathrooms, neighborhood, condition,
                  asking_price, renovation_budget, interest_rate):
    """
    Perform Flip Analysis:
    - FMV = Model prediction
    - Net Profit = FMV - (Asking Price + Renovation Budget)
    - Profit Margin = Net Profit / FMV
    - HIGH RISK if Profit Margin < 15%
    """
    # Validate inputs
    if square_feet <= 0 or asking_price <= 0:
        return "Please enter valid property details", "", "", ""
    
    # Calculate FMV
    fmv = predict_fmv(square_feet, bedrooms, bathrooms, neighborhood, condition)
    
    # Calculate financials
    total_investment = asking_price + renovation_budget
    net_profit = fmv - total_investment
    profit_margin = (net_profit / fmv) * 100 if fmv > 0 else 0
    
    # Format outputs
    fmv_text = f"${fmv:,.2f}"
    net_profit_text = f"${net_profit:,.2f}"
    profit_margin_text = f"{profit_margin:.1f}%"
    
    # Risk assessment
    if profit_margin < 15:
        risk_text = "⚠️ HIGH RISK - Profit margin below 15%"
    else:
        risk_text = "✅ ACCEPTABLE - Profit margin meets threshold"
    
    return fmv_text, net_profit_text, profit_margin_text, risk_text

# =============================================================================
# RENTAL ANALYSIS
# =============================================================================

def rental_analysis(square_feet, bedrooms, bathrooms, neighborhood, condition,
                    asking_price, renovation_budget, interest_rate):
    """
    Perform Rental Analysis:
    - Monthly Rent = 0.8% × FMV
    - Monthly Expenses = 1.5% × Asking Price / 12
    - Mortgage = 30-year fixed formula
    - Cash Flow = Rent - (Mortgage + Expenses)
    - NEGATIVE GEARING if Cash Flow < 0
    """
    # Validate inputs
    if square_feet <= 0 or asking_price <= 0:
        return "Please enter valid property details", "", "", "", ""
    
    # Calculate FMV
    fmv = predict_fmv(square_feet, bedrooms, bathrooms, neighborhood, condition)
    
    # Calculate rental income (0.8% of FMV per month)
    monthly_rent = 0.008 * fmv
    
    # Calculate monthly expenses (1.5% of asking price per year / 12)
    monthly_expenses = (0.015 * asking_price) / 12
    
    # Calculate mortgage (assuming full price financed)
    loan_amount = asking_price + renovation_budget
    monthly_mortgage = calculate_monthly_mortgage(loan_amount, interest_rate)
    
    # Calculate cash flow
    cash_flow = monthly_rent - (monthly_mortgage + monthly_expenses)
    
    # Format outputs
    fmv_text = f"${fmv:,.2f}"
    rent_text = f"${monthly_rent:,.2f}"
    mortgage_text = f"${monthly_mortgage:,.2f}"
    expenses_text = f"${monthly_expenses:,.2f}"
    
    if cash_flow < 0:
        cash_flow_text = f'<span style="color: red; font-weight: bold;">NEGATIVE GEARING: ${cash_flow:,.2f}/month</span>'
    else:
        cash_flow_text = f'<span style="color: green; font-weight: bold;">Positive Cash Flow: ${cash_flow:,.2f}/month</span>'
    
    return fmv_text, rent_text, mortgage_text, expenses_text, cash_flow_text

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Custom CSS for styling
custom_css = """
.risk-warning {
    color: red;
    font-weight: bold;
    font-size: 1.2em;
}
.success-indicator {
    color: green;
    font-weight: bold;
}
"""

with gr.Blocks(title="Real Estate Investor Dashboard", css=custom_css) as demo:
    gr.Markdown("""
    # 🏠 Real Estate Investor Dashboard
    
    Analyze potential real estate investments using machine learning predictions.
    Enter property details and financials to compare **Flip** vs **Rental** strategies.
    
    ---
    """)
    
    # Shared inputs
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Property Details")
            square_feet = gr.Number(label="Square Feet", value=2000, minimum=100, maximum=10000)
            bedrooms = gr.Slider(label="Bedrooms", minimum=1, maximum=10, value=3, step=1)
            bathrooms = gr.Slider(label="Bathrooms", minimum=1, maximum=5, value=2, step=1)
            neighborhood = gr.Dropdown(label="Neighborhood", choices=NEIGHBORHOODS, value="Maplewood")
            condition = gr.Slider(label="Condition (1-10)", minimum=1, maximum=10, value=7, step=1)
        
        with gr.Column():
            gr.Markdown("### 💰 Financial Details")
            asking_price = gr.Number(label="Asking Price ($)", value=350000, minimum=0)
            renovation_budget = gr.Number(label="Renovation Budget ($)", value=50000, minimum=0)
            interest_rate = gr.Number(label="Interest Rate (%)", value=6.5, minimum=0, maximum=20)
    
    gr.Markdown("---")
    
    # Tabs for Flip and Rental Analysis
    with gr.Tabs():
        # FLIP ANALYSIS TAB
        with gr.Tab("🔨 Flip Analysis"):
            gr.Markdown("""
            **Flip Strategy Analysis**
            
            Calculate potential profit from buying, renovating, and selling the property.
            - **Net Profit** = FMV - (Asking Price + Renovation Budget)
            - **Profit Margin** = Net Profit / FMV
            - ⚠️ **HIGH RISK** if Profit Margin < 15%
            """)
            
            flip_btn = gr.Button("Analyze Flip Potential", variant="primary")
            
            with gr.Row():
                flip_fmv = gr.Textbox(label="Fair Market Value (FMV)", interactive=False)
                flip_profit = gr.Textbox(label="Net Profit", interactive=False)
                flip_margin = gr.Textbox(label="Profit Margin", interactive=False)
            
            flip_risk = gr.Textbox(label="Risk Assessment", interactive=False)
            
            flip_btn.click(
                fn=flip_analysis,
                inputs=[square_feet, bedrooms, bathrooms, neighborhood, condition,
                        asking_price, renovation_budget, interest_rate],
                outputs=[flip_fmv, flip_profit, flip_margin, flip_risk]
            )
        
        # RENTAL ANALYSIS TAB
        with gr.Tab("🏢 Rental Analysis"):
            gr.Markdown("""
            **Rental Strategy Analysis**
            
            Calculate potential cash flow from renting the property.
            - **Monthly Rent** = 0.8% × FMV
            - **Monthly Expenses** = 1.5% × Asking Price / 12
            - **Mortgage** = 30-year fixed payment
            - **Cash Flow** = Rent - (Mortgage + Expenses)
            - ⚠️ **NEGATIVE GEARING** warning if Cash Flow < 0
            """)
            
            rental_btn = gr.Button("Analyze Rental Potential", variant="primary")
            
            with gr.Row():
                rental_fmv = gr.Textbox(label="Fair Market Value (FMV)", interactive=False)
                rental_rent = gr.Textbox(label="Monthly Rent", interactive=False)
            
            with gr.Row():
                rental_mortgage = gr.Textbox(label="Monthly Mortgage", interactive=False)
                rental_expenses = gr.Textbox(label="Monthly Expenses", interactive=False)
            
            rental_cashflow = gr.HTML(label="Cash Flow Analysis")
            
            rental_btn.click(
                fn=rental_analysis,
                inputs=[square_feet, bedrooms, bathrooms, neighborhood, condition,
                        asking_price, renovation_budget, interest_rate],
                outputs=[rental_fmv, rental_rent, rental_mortgage, rental_expenses, rental_cashflow]
            )
    
    gr.Markdown("""
    ---
    
    ### 📊 Model Information
    
    - **Prediction Model**: Linear Regression (R² = 0.763)
    - **Features Used**: Square Feet, Bedrooms, Bathrooms, Condition, Effective Age, Neighborhood Tier
    - **Training Data**: 1,988 cleaned real estate transactions
    
    *Note: FMV predictions are estimates based on historical data. Actual market values may vary.*
    """)

# =============================================================================
# LAUNCH
# =============================================================================

if __name__ == "__main__":
    demo.launch(share=False)
