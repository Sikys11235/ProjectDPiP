## Table of Contents
- [Installation](#installation)
- [Description](#description)
- [Next Steps](#future-work)

## Installation
- Install dependencies using `pip install -r requirements.txt`
- Set the correct directory, you can choose two
- Run the script in command line, the script has 3 parameters (stocks, start_date, num_days)
    - One can request a list of stocks by running --list-stocks
- exemplary command is **python run_main.py AAPL V JPM --start-date 2023-01-01 --num-days 10**
    - There are restrictions to the code
        - Number of days should be between 1-60 (period above 60 days became too slow and it is not usual to predict stocks for longer periods)
        - Start date should be after 2010-01-01 (but this can be modified in config.ini by setting start date earlier)
- Keep in mind that choosing more stocks (>3) for more days (>20) leads to longer run-time of the script (> 1 minute)

## Description
- Create a function to download historical stock data for a given ticker
- Allow users to input a stock ticker and specify the date range
- Preprocess the data (e.g., handling missing values, scaling, or feature engineering) before applying any machine learning algorithms
- Split the data into training and testing sets
- Choose a suitable machine learning model for time series forecasting, such as Linear Regression, ARIMA, or LSTM (Long Short-Term Memory) neural networks. Train the model using the training data
- Evaluate the model performance on the testing set using appropriate metrics, such as Mean Absolute Error (MAE) or Mean Squared Error (MSE)
- Use the trained model to forecast future stock returns for the specified stock
- Visualise our predictions

## Next Steps (what is not - but could have been implemented to make the project (even) better)
- Adding more models, some with hyperparameters (Ridge, Lasso regression with alpha set in the config) - this caused version errors that we were unable to fix in the given time
- More visualization options (colors of the bar charts, line charts) - but this is not too important
- Comparison with (if applicable) actual returns of the stocks if past period was chosen
- Further data analysis of the predicted returns, f.e. storing them in a separate csv + histogram and segmented by stock to see how they accumulate over time, if there is a trend when increasing the sample size of predictions - this was initially in the ipynb file, but was not segmented by stock and it is not the main part of the scope
