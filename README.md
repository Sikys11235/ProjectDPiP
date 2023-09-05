
Current to do's:
- vytvořit soubor requirements.txt - všechny packages (Vojta)
- vytvořit soubor config (to ale nevím co by mělo dělat)
- přidat možnost aby šlo vybrat více stocks (Vojta)
- User interface - vyskakovací okno, kde by se všechno navolilo (stock/s, datum start, datum konec, granularita??)
- Něco ještě přidat?
- Proof-read code, občas tam jsou defaultní názvy funkcí, vyhodit některé naše komentáře ##

Fungování:
User vybere datum, stock, sesbíráme data z yfinance pro ten rok a vyprodukujeme model (x kroků) a pak uděláme vizializaci


Napadlo mě přidat něco jakože choose your stock (kdybychom jich měli víc) podle nějakých inputů (třeba volatilita, zhodnocení,...) a my bychom na konci řekli ok, vezmeme stock XY a vydělá se na tom XY podle našeho forecastu. Nebo aby tam byla nějaká přidaná hodnota, než jen ok tohle jsou stocks a tohle je jejich forecast (returns)


To start this project, I recommend using the yfinance package, as it simplifies the process of fetching data from Yahoo Finance. Here's an outline of the steps you can follow to build your stock price prediction project:

- Create a function to download historical stock data for a given ticker
- Allow users to input a stock ticker and specify the date range:
- Preprocess the data (e.g., handling missing values, scaling, or feature engineering) before applying any machine learning algorithms.
- Split the data into training and testing sets
- Choose a suitable machine learning model for time series forecasting, such as Linear Regression, ARIMA, or LSTM (Long Short-Term Memory) neural networks. Train the model using the training data.
- Evaluate the model performance on the testing set using appropriate metrics, such as Mean Absolute Error (MAE) or Mean Squared Error (MSE)
- Use the trained model to forecast future stock returns for the specified stock
- Visualise our predictions
