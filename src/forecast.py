import pandas as pd
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np

# Relative imports
from . import charts_reports_path, suppress_statsmodels_warnings

forecast_horizon = 3
n_splits = 3
mae_list = []
mape_list = []


preprocessed_df = pd.read_csv(charts_reports_path / "preprocessed_royal_perth_data.csv", 
                            parse_dates=['Date'])

min_date = preprocessed_df['Date'].min()
max_date = preprocessed_df['Date'].max()
preprocessed_df.index = pd.date_range(start=min_date, 
                          end=max_date, 
                          freq='D')
preprocessed_df = preprocessed_df.set_index('Date', 
                                            verify_integrity=True)

# Perform median value imputation
preprocessed_df = preprocessed_df.fillna(preprocessed_df[['Attendance', 'Admissions', 'Tri_1', 'Tri_2', 'Tri_3', 'Tri_4',
       'Tri_5']].median())


# Rolling-origin cross-validation loop
@suppress_statsmodels_warnings
def calculate_cross_validation(df):
       for i in range(n_splits):
              train_size = len(df) - forecast_horizon * (n_splits - i)
              train_data = df.iloc[:train_size]['Tri_1']
              test_data = df.iloc[train_size:train_size + forecast_horizon]['Tri_1']
              
              stlf = STLForecast(train_data, 
                                 ARIMA, 
                                 model_kwargs={"order": (2, 1, 0)}, 
                                 robust=True, 
                                 seasonal=13)
              stlf_result = stlf.fit()
              forecast = stlf_result.forecast(steps=forecast_horizon)
              
              # Calculate errors
              mae = mean_absolute_error(test_data, forecast)
              mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
              
              mae_list.append(mae)
              mape_list.append(mape)

       # Since there are 3-folds in the train/test split, calculate the average of the error across 3-folds
       mean_mae = np.mean(mae_list)
       mean_mape = np.mean(mape_list)
       print(f'Cross-validated Mean Absolute Error (MAE): {mean_mae:.2f}')
       print(f'Cross-validated Mean Absolute Percentage Error (MAPE): {mean_mape:.2f}%')


@suppress_statsmodels_warnings
def train_forecast_model(df):
       stlf = STLForecast(df['Tri_1'], 
                     ARIMA, 
                     model_kwargs={"order": (2, 1, 0)},
                     robust=True,
                     seasonal=13)  # Adjust `seasonal` based on periodicity

       # Fit the model
       stlf_result = stlf.fit()
       # Print summary of the fitted model
       print(stlf_result.summary())

       # Generate 3-day forecast
       forecast = stlf_result.forecast(steps=forecast_horizon)
       plt.figure(figsize=(20,6))
       plt.plot(df['Tri_1'])
       plt.plot(forecast)
       plt.title("Forecast Model for Royal Perth ED Triage Category 1")
       plt.savefig(charts_reports_path / "forecast_model_plot.png", 
              bbox_inches='tight')


calculate_cross_validation(df=preprocessed_df)
train_forecast_model(df=preprocessed_df)