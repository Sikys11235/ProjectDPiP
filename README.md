
*Read.me will be updaated and cleaned during final submission*

Current to do's:
- vytvořit soubor requirements.txt - všechny packages (Vojta)
- vytvořit soubor config ?
- přidat možnost aby šlo vybrat více stocks (Vojta)
- User interface - parse
- Něco ještě přidat?
- Proof-read code, občas tam jsou defaultní názvy funkcí, vyhodit některé naše komentáře ##

To start this project, I recommend using the yfinance package, as it simplifies the process of fetching data from Yahoo Finance. Here's an outline of the steps you can follow to build your stock price prediction project:

1) requirements.txt contains all the packages necessary
2) run_main.py is the script to be run
3) other files are making sure run_main works properly

- Create a function to download historical stock data for a given ticker
- Allow users to input a stock ticker and specify the date range:
- Preprocess the data (e.g., handling missing values, scaling, or feature engineering) before applying any machine learning algorithms.
- Split the data into training and testing sets
- Choose a suitable machine learning model for time series forecasting, such as Linear Regression, ARIMA, or LSTM (Long Short-Term Memory) neural networks. Train the model using the training data.
- Evaluate the model performance on the testing set using appropriate metrics, such as Mean Absolute Error (MAE) or Mean Squared Error (MSE)
- Use the trained model to forecast future stock returns for the specified stock
- Visualise our predictions
