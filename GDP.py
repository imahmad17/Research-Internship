import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ARIMA
import urllib.request
from urllib.error import URLError

github_path = 'https://raw.githubusercontent.com/imahmad17/Research-Internship/main/world%20gdp%20dataset.csv'
data = pd.read_csv(github_path)

# Rename the column
data.rename(columns={'GDP, current prices(Billions of U.S. dollars)': 'countries'}, inplace=True)

# Set the 'country' column as the index
data.set_index('countries', inplace=True)

# Transpose the data to have years as rows and countries as columns
transposed_data = data.transpose()

# Convert the index to datetime format
transposed_data.index = pd.to_datetime(transposed_data.index, format='%Y')

all_countries = transposed_data.columns

predicted_gdp = {}

for country in all_countries:
    country_data = transposed_data[country].dropna()

    forecast_steps =11
    d=1
    
    last_year = country_data.index[-1].year
    model = ARIMA(country_data, order=(0, d, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Predict GDP for 2024
    gdp_2024 = forecast[-1]
    predicted_gdp[country] = gdp_2024

# Plot the predicted GDP values for all countries in 2024
plt.figure(figsize=(12, 8))
plt.bar(predicted_gdp.keys(), predicted_gdp.values(), color='blue')
plt.title('Predicted GDP for All Countries in 2024')
plt.xlabel('Country')
plt.ylabel('GDP in 2024')
plt.xticks(rotation=90, fontsize=4)
plt.tight_layout()
plt.show()

# Get user input for the country
user_country = input("Enter the country name (first letter capital): ")

if user_country in transposed_data.columns:
    country_data = transposed_data[user_country].dropna()
    
    # Forecast future GDP values for the chosen country
    forecast_steps = 11  # Extend forecast to 2024
    d = 1  # Differencing parameter
    
    last_year = country_data.index[-1].year
    model = ARIMA(country_data, order=(0, d, 0))  # Start with (0, d, 0) and let auto ARIMA find p and q
    model_fit = model.fit() 
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Create year index for the forecasted values
    forecast_years = [last_year + i for i in range(1, forecast_steps+1)]
    
    # Plot the forecasted GDP values for the chosen country
    plt.figure(figsize=(10, 6))
    plt.plot(country_data.index, country_data, label='Historical Data')
    plt.plot(forecast_years[:-1], forecast[:-1], label='Forecasted Data', color='red')  # Adjust indexing here
    plt.title(f'GDP Forecast for {user_country}')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.legend()
    plt.show()

    # Predict GDP for 2024
    gdp_2024 = forecast[-1]
    print(f'Predicted GDP for {user_country} in 2024: {gdp_2024}')
else:
    print("Country not found in the dataset.")
