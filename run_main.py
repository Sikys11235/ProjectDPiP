#Requirements

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

#Popular stocks

stock_info = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com Inc.',
    'FB': 'Meta Platforms Inc. (Facebook)',
    'TSLA': 'Tesla Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'JNJ': 'Johnson & Johnson',
    'V': 'Visa Inc.',
    'PG': 'Procter & Gamble Co.',
    'NVDA': 'NVIDIA Corporation',
    'BRK-B': 'Berkshire Hathaway Inc.',
    'UNH': 'UnitedHealth Group Incorporated',
    'DIS': 'The Walt Disney Company',
    'MA': 'Mastercard Incorporated',
    'PYPL': 'PayPal Holdings Inc.',
    'HD': 'The Home Depot Inc.',
    'CMCSA': 'Comcast Corporation',
    'VZ': 'Verizon Communications Inc.',
    'INTC': 'Intel Corporation',
    'T': 'AT&T Inc.',
    'PFE': 'Pfizer Inc.',
    'MRK': 'Merck & Co. Inc.',
    'NFLX': 'Netflix Inc.',
    'PEP': 'PepsiCo Inc.',
    'WMT': 'Walmart Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'ADBE': 'Adobe Inc.',
    'BAC': 'Bank of America Corporation',
    'KO': 'The Coca-Cola Company',
    'XOM': 'Exxon Mobil Corporation',
    'GOOG': 'Alphabet Inc. (Google)',
    'CRM': 'Salesforce.com Inc.',
    'ABBV': 'AbbVie Inc.',
    'NKE': 'NIKE Inc.',
    'ABT': 'Abbott Laboratories',
    'MO': 'Altria Group Inc.',
    'LLY': 'Eli Lilly and Company',
    'PEP': 'PepsiCo Inc.',
    'CVX': 'Chevron Corporation',
    'BA': 'The Boeing Company',
    'CAT': 'Caterpillar Inc.',
    'COST': 'Costco Wholesale Corporation',
}

#Interface parser

custom_help = """
Stock Prediction Tool

If you want to see the complete list of stocks available with the description, use the following code:
python run_main.py dummy --list-stocks --start-date dummy
"""

parser = argparse.ArgumentParser(description=custom_help, formatter_class=argparse.RawDescriptionHelpFormatter)


parser.add_argument('stocks', nargs='+', type=str, help='Required stock symbol(s)')
parser.add_argument('--start-date', type=str, required=True, help='Start date for prediction (YYYY-MM-DD)')
parser.add_argument('--num-days', type=int, default=5, help='Number of days to predict (default: 5) - doesnt display weekends')
parser.add_argument('--list-stocks', action='store_true', help='List available stocks and exit')

args = parser.parse_args()

if args.list_stocks:
    print("Available Stocks:")
    for symbol, description in stock_info.items():
        print(f"{symbol}: {description}")
    exit(0)

stock_symbols = args.stocks
end_date = args.start_date
num_days = args.num_days

# Validate selected stocks
for stock_symbol in stock_symbols:
    if stock_symbol not in stock_info:
        print(f"Invalid stock symbol: {stock_symbol}")
        exit(1)

# Function to fetch stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data.dropna()

# Function to calculate daily returns
def calculate_daily_returns(stock_data):
    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
    stock_data = stock_data.dropna()  # Remove rows with NaN values
    return stock_data

# Function to calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA) - not sure if needed actually
def calculate_sma_ema(stock_data, window):
    stock_data = stock_data.copy()  # Create a copy of the DataFrame to work on
    stock_data.loc[:, 'SMA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data.loc[:, 'EMA'] = stock_data['Close'].ewm(span=window, adjust=False).mean()
    return stock_data.dropna()

# Function to prepare the data
def prepare_data(stock_data, num_days):
    X = []
    y = []
    
    for i in range(len(stock_data) - num_days):
        X.append(stock_data[['Daily_Return', 'SMA', 'EMA']].iloc[i:i+num_days].values.flatten())
        y.append(stock_data['Daily_Return'].iloc[i+num_days])
           
    return np.array(X), np.array(y)

# Function to train and evaluate machine learning models - add ARIMA?
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
    }
    
    best_model = None
    best_mse = float('inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
            
        mse = mean_squared_error(y_test, y_pred)
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
            print(f"Best Model: {name}")
    
    print(f"Best MSE: {best_mse}")
    
    return best_model

# Function to visualize predictions and daily returns
def visualize_predictions(predicted_returns, date_labels, stock_symbol):
    plt.style.use('dark_background')  # Set the style to dark background
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the bar plot
    bars = ax.bar(date_labels, predicted_returns, label=f'{stock_symbol} Predicted Returns', color=np.random.rand(3,), edgecolor='grey')

    # Add title and labels
    ax.set_title(f'{stock_symbol} Predicted Stock Returns', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Predicted Returns', color='white')

    # Set the tick parameters to be white
    ax.tick_params(axis='both', colors='white')

    # Add grid and set its color to mild dark grey
    ax.grid(True, color='#555555')

    # Add legend with white text
    ax.legend(loc='upper left', frameon=True, facecolor='#363636', edgecolor='white', framealpha=1, fontsize=12, labelcolor='white')

    # Outline in mild dark grey
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')

    plt.show()

if __name__ == "__main__":

    if not stock_symbols:
        print("No stock symbols provided. Exiting.")
        exit()
    
    try:
        num_days = int(num_days)
        if not (1 <= num_days <= 60):
            print("Invalid number of days. Please enter a value between 1 and 60.")
            exit()
    except ValueError:
        print("Invalid number of days. Please enter a valid number.")
        exit()

    start_date = "2010-01-01"  # dynamic start date caused problems

    predicted_returns_dict = {}

    for stock_symbol in stock_symbols:
        stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
        
        if len(stock_data) <= num_days:
            print(f"Not enough data for stock symbol {stock_symbol}. Skipping.")
            continue
        
        daily_returns_data = calculate_daily_returns(stock_data)
        
        window = 20
        stock_data = calculate_sma_ema(daily_returns_data, window)
        
        X, y = prepare_data(stock_data, num_days)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        best_model = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        
        last_week_data = X_test[-1].reshape(1, -1)
        
        predicted_returns = []

        for i in range(num_days):
            prediction = best_model.predict(last_week_data)
            predicted_returns.append(prediction[0])
            new_data_point = np.array([prediction[0], stock_data['SMA'].iloc[-num_days+i+1], stock_data['EMA'].iloc[-num_days+i+1]])
            last_week_data = np.roll(last_week_data, -3)
            last_week_data[0, -3:] = new_data_point

        last_date_in_data = stock_data.index[-1]
        prediction_dates = pd.date_range(start=last_date_in_data, periods=num_days+1, freq='B')[1:]  # Skip the first date to start from the next business day
        
        predicted_returns_dict[stock_symbol] = predicted_returns

        visualize_predictions(predicted_returns, prediction_dates, stock_symbol)

    plt.figure(figsize=(12, 6))

    for stock_symbol, predicted_returns in predicted_returns_dict.items():
        plt.plot(prediction_dates, predicted_returns, label=f'{stock_symbol} Predicted Returns', marker='o')

    plt.title('Predicted Stock Returns')
    plt.xlabel('Date')
    plt.ylabel('Predicted Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    
