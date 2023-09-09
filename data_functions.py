import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
import argparse

# Import models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Function to fetch stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance for a given stock symbol and date range.

    Parameters:
        stock_symbol (str): The stock symbol to fetch data for.
        start_date (str): The starting date for the data in the format YYYY-MM-DD.
        end_date (str): The ending date for the data in the format YYYY-MM-DD.
    
    Returns:
        pd.DataFrame: Data frame containing stock data.
    """
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data.dropna()

# Function to calculate daily returns
def calculate_daily_returns(stock_data):
    """
    Calculate daily returns for stock data.

    Parameters:
        stock_data (pd.DataFrame): Data frame containing stock data.
    
    Returns:
        pd.DataFrame: Data frame with a new column 'Daily_Return' containing daily returns.
    """
    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
    return stock_data.dropna()

# Function to calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)
def calculate_sma_ema(stock_data, window):
    """
    Calculate Simple Moving Average and Exponential Moving Average.

    Parameters:
        stock_data (pd.DataFrame): Data frame containing stock data.
        window (int): The window size for calculating moving averages.
    
    Returns:
        pd.DataFrame: Data frame with new columns 'SMA' and 'EMA' containing the calculated averages.
    """
    stock_data = stock_data.copy()
    stock_data.loc[:, 'SMA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data.loc[:, 'EMA'] = stock_data['Close'].ewm(span=window, adjust=False).mean()
    return stock_data.dropna()

# Function to prepare the data
def prepare_data(stock_data, num_days):
    """
    Prepare feature and target data for machine learning models.

    Parameters:
        stock_data (pd.DataFrame): Data frame containing stock data.
        num_days (int): Number of days to consider for features.
    
    Returns:
        np.array: Feature data.
        np.array: Target data.
    """
    X = []
    y = []
    for i in range(len(stock_data) - num_days):
        X.append(stock_data[['Daily_Return', 'SMA', 'EMA']].iloc[i:i+num_days].values.flatten())
        y.append(stock_data['Daily_Return'].iloc[i+num_days])
    return np.array(X), np.array(y)

# Function to train and evaluate machine learning models
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate machine learning models.

    Parameters:
        X_train (np.array): Training feature data.
        y_train (np.array): Training target data.
        X_test (np.array): Testing feature data.
        y_test (np.array): Testing target data.
    
    Returns:
        sklearn.model: Best performing model based on MSE.
    """
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Decision Tree': DecisionTreeRegressor()}

    best_model = None
    best_mse = float('inf')
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = model
    return best_model

# Function to visualize predictions and daily returns
def visualize_predictions(predicted_returns, date_labels, stock_symbol):
    """
    Visualize predicted stock returns.

    Parameters:
        predicted_returns (list): List of predicted returns.
        date_labels (list): List of date labels.
        stock_symbol (str): Stock symbol.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(date_labels, predicted_returns, label=f'{stock_symbol} Predicted Returns', color=np.random.rand(3,), edgecolor='grey')
    ax.set_title(f'{stock_symbol} Predicted Stock Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Returns')
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, color='#555555')
    ax.legend(loc='upper left', frameon=True, facecolor='#363636', edgecolor='white', framealpha=1, fontsize=12, labelcolor='white')
    plt.show()
