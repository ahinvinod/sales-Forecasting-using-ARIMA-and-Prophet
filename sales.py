# Sales Forecasting using ARIMA and Prophet
# Author: Ahin Vinod
# Project: Time-Series Analysis for Sales Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Install required packages (uncomment if needed):
# pip install pandas numpy matplotlib seaborn statsmodels prophet scikit-learn

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*60)
print("SALES FORECASTING PROJECT")
print("Time-Series Analysis using ARIMA and Prophet")
print("="*60)

# ============================================
# 1. DATA GENERATION (Sample Sales Data)
# ============================================
print("\n1. Generating Sample Sales Data...")

# Create sample sales data with trend and seasonality
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', end='2024-10-31', freq='D')
n = len(date_range)

# Base trend + seasonality + noise
trend = np.linspace(1000, 5000, n)
seasonal = 500 * np.sin(np.arange(n) * 2 * np.pi / 365)
noise = np.random.normal(0, 200, n)
sales = trend + seasonal + noise

# Create DataFrame
df = pd.DataFrame({
    'Date': date_range,
    'Sales': sales
})

print(f"Dataset created: {len(df)} days of sales data")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================
print("\n2. Exploratory Data Analysis...")

# Basic statistics
print("\nSales Statistics:")
print(df['Sales'].describe())

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Plot the time series
plt.figure(figsize=(15, 5))
plt.plot(df['Date'], df['Sales'], linewidth=1)
plt.title('Historical Sales Data', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sales_timeseries.png', dpi=300)
print("✓ Time series plot saved as 'sales_timeseries.png'")

# ============================================
# 3. STATIONARITY TEST
# ============================================
print("\n3. Testing for Stationarity (ADF Test)...")

def adf_test(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("✓ Data is STATIONARY (p-value <= 0.05)")
        return True
    else:
        print("✗ Data is NON-STATIONARY (p-value > 0.05)")
        return False

is_stationary = adf_test(df['Sales'])

# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================
print("\n4. Splitting Data into Train and Test Sets...")

train_size = int(len(df) * 0.8)
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

print(f"Training set: {len(train_df)} observations")
print(f"Test set: {len(test_df)} observations")

# ============================================
# 5. ARIMA MODEL
# ============================================
print("\n5. Building ARIMA Model...")

# Fit ARIMA model (p, d, q) = (5, 1, 2)
# You can tune these parameters based on ACF/PACF plots
arima_model = ARIMA(train_df['Sales'], order=(5, 1, 2))
arima_fit = arima_model.fit()

print("\nARIMA Model Summary:")
print(arima_fit.summary())

# Make predictions
arima_pred = arima_fit.forecast(steps=len(test_df))
arima_pred_df = pd.DataFrame({
    'Date': test_df['Date'].values,
    'Actual': test_df['Sales'].values,
    'ARIMA_Predicted': arima_pred
})

# Calculate metrics
arima_mae = mean_absolute_error(test_df['Sales'], arima_pred)
arima_rmse = np.sqrt(mean_squared_error(test_df['Sales'], arima_pred))
arima_r2 = r2_score(test_df['Sales'], arima_pred)

print(f"\nARIMA Model Performance:")
print(f"MAE: {arima_mae:.2f}")
print(f"RMSE: {arima_rmse:.2f}")
print(f"R² Score: {arima_r2:.4f}")

# ============================================
# 6. PROPHET MODEL
# ============================================
print("\n6. Building Prophet Model...")

# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_train = train_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
prophet_test = test_df.rename(columns={'Date': 'ds', 'Sales': 'y'})

# Initialize and fit Prophet model
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
prophet_model.fit(prophet_train)

# Make future dataframe for predictions
future = prophet_model.make_future_dataframe(periods=len(test_df))
prophet_forecast = prophet_model.predict(future)

# Extract test predictions
prophet_pred = prophet_forecast.tail(len(test_df))['yhat'].values

prophet_pred_df = pd.DataFrame({
    'Date': test_df['Date'].values,
    'Actual': test_df['Sales'].values,
    'Prophet_Predicted': prophet_pred
})

# Calculate metrics
prophet_mae = mean_absolute_error(test_df['Sales'], prophet_pred)
prophet_rmse = np.sqrt(mean_squared_error(test_df['Sales'], prophet_pred))
prophet_r2 = r2_score(test_df['Sales'], prophet_pred)

print(f"\nProphet Model Performance:")
print(f"MAE: {prophet_mae:.2f}")
print(f"RMSE: {prophet_rmse:.2f}")
print(f"R² Score: {prophet_r2:.4f}")

# ============================================
# 7. MODEL COMPARISON
# ============================================
print("\n7. Model Comparison...")

comparison = pd.DataFrame({
    'Model': ['ARIMA', 'Prophet'],
    'MAE': [arima_mae, prophet_mae],
    'RMSE': [arima_rmse, prophet_rmse],
    'R² Score': [arima_r2, prophet_r2]
})

print("\n" + comparison.to_string(index=False))

# Determine best model
best_model = comparison.loc[comparison['RMSE'].idxmin(), 'Model']
print(f"\n✓ Best Model: {best_model} (lowest RMSE)")

# ============================================
# 8. VISUALIZATION
# ============================================
print("\n8. Creating Visualizations...")

# Plot both predictions
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# ARIMA predictions
axes[0].plot(test_df['Date'], test_df['Sales'], label='Actual', linewidth=2, color='black')
axes[0].plot(test_df['Date'], arima_pred, label='ARIMA Predicted', linewidth=2, color='red', linestyle='--')
axes[0].set_title('ARIMA Model Predictions', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Sales')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Prophet predictions
axes[1].plot(test_df['Date'], test_df['Sales'], label='Actual', linewidth=2, color='black')
axes[1].plot(test_df['Date'], prophet_pred, label='Prophet Predicted', linewidth=2, color='blue', linestyle='--')
axes[1].set_title('Prophet Model Predictions', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Sales')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300)
print("✓ Model predictions plot saved as 'model_predictions.png'")

# Combined comparison plot
plt.figure(figsize=(15, 6))
plt.plot(test_df['Date'], test_df['Sales'], label='Actual', linewidth=2, color='black')
plt.plot(test_df['Date'], arima_pred, label='ARIMA', linewidth=2, color='red', linestyle='--', alpha=0.7)
plt.plot(test_df['Date'], prophet_pred, label='Prophet', linewidth=2, color='blue', linestyle='--', alpha=0.7)
plt.title('Model Comparison: Actual vs Predictions', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('combined_predictions.png', dpi=300)
print("✓ Combined predictions plot saved as 'combined_predictions.png'")

# ============================================
# 9. FUTURE FORECASTING
# ============================================
print("\n9. Forecasting Future Sales (Next 90 days)...")

# Retrain on full dataset
full_arima = ARIMA(df['Sales'], order=(5, 1, 2)).fit()
future_arima = full_arima.forecast(steps=90)

# Prophet future forecast
prophet_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_full.fit(df.rename(columns={'Date': 'ds', 'Sales': 'y'}))
future_dates = prophet_full.make_future_dataframe(periods=90)
future_prophet = prophet_full.predict(future_dates)

# Create future forecast dataframe
future_df = pd.DataFrame({
    'Date': pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=90),
    'ARIMA_Forecast': future_arima,
    'Prophet_Forecast': future_prophet.tail(90)['yhat'].values
})

print("\nFuture Sales Forecast (First 10 days):")
print(future_df.head(10))

# ============================================
# 10. EXPORT DATA FOR TABLEAU
# ============================================
print("\n10. Exporting Data for Tableau Dashboard...")

# Combine historical and forecasted data
tableau_data = pd.concat([
    df.assign(Type='Historical', ARIMA_Forecast=np.nan, Prophet_Forecast=np.nan),
    future_df.assign(Type='Forecast', Sales=np.nan)
], ignore_index=True)

tableau_data.to_csv('sales_forecast_tableau.csv', index=False)
print("✓ Data exported to 'sales_forecast_tableau.csv'")

# Export model metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R² Score'],
    'ARIMA': [arima_mae, arima_rmse, arima_r2],
    'Prophet': [prophet_mae, prophet_rmse, prophet_r2]
})
metrics_df.to_csv('model_metrics.csv', index=False)
print("✓ Model metrics exported to 'model_metrics.csv'")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nFiles Generated:")
print("1. sales_timeseries.png - Historical sales visualization")
print("2. model_predictions.png - Individual model predictions")
print("3. combined_predictions.png - Comparative predictions")
print("4. sales_forecast_tableau.csv - Data for Tableau dashboard")
print("5. model_metrics.csv - Model performance metrics")

print("\n" + "="*60)
print("TABLEAU DASHBOARD INSTRUCTIONS")
print("="*60)
print("""
To create an interactive Tableau dashboard:

1. IMPORT DATA:
   - Open Tableau Desktop
   - Connect to 'sales_forecast_tableau.csv'
   
2. CREATE WORKSHEETS:
   
   a) Historical Sales Trend:
      - Drag Date to Columns
      - Drag Sales to Rows
      - Filter by Type = 'Historical'
      - Add trend line
      
   b) Forecast Comparison:
      - Drag Date to Columns
      - Drag Sales, ARIMA_Forecast, Prophet_Forecast to Rows
      - Use dual axis
      - Add color coding for different lines
      
   c) Model Accuracy (import model_metrics.csv):
      - Create bar chart comparing MAE, RMSE, R²
      - Use color for different models
      
   d) Monthly Aggregation:
      - Convert Date to Month
      - Show SUM(Sales) by month
      - Add Year filter
      
3. BUILD DASHBOARD:
   - Combine all worksheets
   - Add filters: Date Range, Model Type
   - Add KPIs: Total Sales, Avg Sales, Growth Rate
   - Add text boxes for insights
   - Format with professional color scheme

4. PUBLISH:
   - Save as .twbx file
   - Publish to Tableau Server/Public if needed
""")

print("\n✓ Ready to build your Tableau dashboard!")
print("="*60)