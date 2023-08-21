# ProjectDPiP
Repository for our DPiP project

#Project Structure
- Download stock data from Kaggle
- Clean, manipulate and visualize the respective stock prices
- forecast stock prices using econometric techniques haha



#Purpose of this project
Napadlo mě přidat něco jakože choose your stock (kdybychom jich měli víc) podle nějakých inputů (třeba volatilita, zhodnocení,...) a my bychom na konci řekli ok, vezmeme stock XY a vydělá se na tom XY podle našeho forecastu. Nebo aby tam byla nějaká přidaná hodnota, než jen ok tohle jsou stocks a tohle je jejich forecast. Fanda má něco podobného, že inputne počasí a jejich kód jim nabídne zdroj.


To start this project, I recommend using the `yfinance` package, as it simplifies the process of fetching data from Yahoo Finance. Here's an outline of the steps you can follow to build your stock price prediction project:


# steps
1. Install the `yfinance` package:

2. Import the required libraries:

3. Create a function to download historical stock data for a given ticker(s):

4. Allow users to input a stock ticker and specify the date range:

5. Preprocess the data (e.g., handling missing values, scaling, or feature engineering) before applying any machine learning algorithms.

6. Split the data into training and testing sets.

7. Choose a suitable machine learning model for time series forecasting, such as Linear Regression, ARIMA, or LSTM (Long Short-Term Memory) neural networks. Train the model using the training data.

8. Evaluate the model performance on the testing set using appropriate metrics, such as Mean Absolute Error (MAE) or Mean Squared Error (MSE).

9. Use the trained model to forecast future stock returns for the specified stock.

10. Visualise our predictions.

