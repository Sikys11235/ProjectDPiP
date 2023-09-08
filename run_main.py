# Importing functions from another Python file
from data_functions import fetch_stock_data, calculate_daily_returns, calculate_sma_ema, prepare_data, train_and_evaluate_models, visualize_predictions

# Importing settings from config.ini
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# Reading visualization settings
viz_config = config['Visualization']

figure_size = tuple(map(int, viz_config['figure_size'].split(',')))
title = viz_config['title']
x_label = viz_config['x_label']
y_label = viz_config['y_label']
show_legend = viz_config.getboolean('show_legend')  # Automatically converts the ini string to a Python boolean
show_grid = viz_config.getboolean('show_grid') # Automatically converts the ini string to a Python boolean

# Requirements
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from configparser import ConfigParser

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

#Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your custom help text", formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('stocks', nargs='+', type=str, help='Required stock symbol(s)')
    parser.add_argument('--start-date', type=str, required=True, help='Start date for prediction (YYYY-MM-DD), please input no earlier date than 2010-01-01')
    parser.add_argument('--num-days', type=int, default=5, help='Number of days to predict (choose a number between 1-60) - doesnt display weekends')
    parser.add_argument('--list-stocks', action='store_true', help='List available stocks and exit')

    args = parser.parse_args()

    if args.list_stocks:
        for symbol, description in stock_info.items():
            print(f"{symbol}: {description}")
        exit(0)

    stock_symbols = args.stocks
    end_date = args.start_date
    num_days = args.num_days
    end_date_str = args.start_date 
    
    # Validate selected stocks
    for stock_symbol in stock_symbols:
        if stock_symbol not in stock_info:
            print(f"Invalid stock symbol: {stock_symbol}")
            exit(1)

    
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
    
    start_date = config['DEFAULT']['start_date']
    predicted_returns_dict = {}
    
    #Validate if correct date was inputed
    
    try:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Please enter a date in the format YYYY-MM-DD.")
        exit(1)

    # Check if the end_date is after 2010-01-01
    
    if end_date < datetime(2010, 1, 1):
        print("Start date for prediction should be no earlier than 2010-01-01.")
        exit(1)
        
    # Stock data fetching

    for stock_symbol in stock_symbols:
        stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
        
        if len(stock_data) <= num_days:
            print(f"Not enough data for stock symbol {stock_symbol}. Skipping.")
            continue
        
        # Model training and performance comparison
        
        daily_returns_data = calculate_daily_returns(stock_data)
        
        window = int(config['DEFAULT']['window_MA'])
        stock_data = calculate_sma_ema(daily_returns_data, window)
        
        X, y = prepare_data(stock_data, num_days)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Shuffle false as the order is important
        
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
        prediction_dates = pd.date_range(start=last_date_in_data, periods=num_days+1, freq='B')[1:] 
        
        predicted_returns_dict[stock_symbol] = predicted_returns

        visualize_predictions(predicted_returns, prediction_dates, stock_symbol)

    plt.figure(figsize=figure_size)

    for stock_symbol, predicted_returns in predicted_returns_dict.items():
        plt.plot(prediction_dates, predicted_returns, label=f'{stock_symbol} Predicted Returns', marker='o')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if show_legend:
        plt.legend()

    if show_grid:
        plt.grid(True) 
    plt.show()    
