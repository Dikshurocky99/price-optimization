import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# === Step 1: Load Data ===
sales_df = pd.read_csv("product_sales.csv", parse_dates=['Date'])
price_demand_df = pd.read_csv("price_demand.csv")
competitor_df = pd.read_csv("competitor_prices.csv")

# === Step 2: Analyze Demand Trend ===
product_name = 'Product A'
product_sales = sales_df[sales_df['Product'] == product_name].set_index('Date')['UnitsSold']

# Forecasting demand using ARIMA
model = ARIMA(product_sales, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=7)

# === Step 3: Visualize Demand Forecast ===
plt.figure(figsize=(10, 4))
plt.plot(product_sales, label='Historical Demand')
plt.plot(forecast.index, forecast, label='Forecasted Demand', linestyle='--')
plt.title(f'Demand Forecast for {product_name}')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# === Step 4: Price Optimization ===
df = price_demand_df[price_demand_df['Product'] == product_name]
X = df[['Price']]
y = df['UnitsSold']

model = LinearRegression()
model.fit(X, y)

# Predict demand and revenue
price_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
predicted_units = model.predict(price_range)
revenue = price_range.flatten() * predicted_units
optimal_index = np.argmax(revenue)
optimal_price = price_range[optimal_index][0]

# === Step 5: Visualize Revenue Curve ===
plt.figure(figsize=(10, 4))
plt.plot(price_range, revenue, label='Revenue')
plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.2f}')
plt.title('Revenue vs Price')
plt.xlabel('Price')
plt.ylabel('Revenue')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Step 6: Competitor Price Analysis ===
plt.figure(figsize=(8, 4))
sns.boxplot(data=competitor_df, x='Product', y='CompetitorPrice')
plt.title('Competitor Price Comparison')
plt.tight_layout()
plt.grid()
plt.show()

# === Step 7: Export to Excel ===
output_df = pd.DataFrame({
    'Price': price_range.flatten(),
    'PredictedUnits': predicted_units,
    'Revenue': revenue
})
output_df['OptimalPrice'] = optimal_price
output_df.to_excel("price_analysis_output.xlsx", index=False)

# Also export forecast
forecast_df = forecast.reset_index()
forecast_df.columns = ['Date', 'ForecastedUnits']
forecast_df.to_excel("forecast_output.xlsx", index=False)

print(f"Suggested Optimal Price for {product_name}: ${optimal_price:.2f}")
