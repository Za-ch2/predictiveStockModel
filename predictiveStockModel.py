import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
'''
# on average there are 252 trading days in a year
NUM_TRADING_DAYS = 252
# we will generate random w (different portfolios)
NUM_PORTFOLIOS = 10000
'''
# create a global list of stocks
stocks = [x.upper() for x in ["AAPL", "ABBV", "ADBE", "AMZN", "bac", "brk.a", "Cat", "crm", "csco", "ctxs", "ddog",
                            "eqix", "googl", "hd", "hon", "ibm", "jnj", "ma", "mdb", "msft", "net", "now", "okta", "orcl",
                            "pfe", "pypl", "shop", "snow", "spot", "sq", "tsla", "twlo", "txn", "u", "unh", "v", "wba", "wdc",
                            "wfc", "wmt", "xlnx", "zm"]]


# historical data - define START and END dates
start_date = '2010-01-01'
end_date = '2020-01-01'

def download_data():
    # name of the stock (key) - stock values (2010-1017) as the values
    stock_data = {}

    for stock in stocks:
        try:
            # closing prices
            ticker = yf.Ticker(stock)
            data = ticker.history(start="2010-01-01")

            # Get all the data if there is no data for 2021
            if not data.index.contains(2021):
                data = ticker.history()

            # Set the start date to the earliest date in the data
            start_date = data.index[0]

            data = ticker.history(start=start_date)

            stock_data[stock] = data['Close']
        except Exception as e:
            print(f"Error fetching data for {stock}: {str(e)}")

    return pd.DataFrame(stock_data)

def test_data():
    # name of the stock (key) - stock values (2021-2022) as the values
    test_data = {}

    for stock in stocks:
        try:
            # closing prices
            ticker = yf.Ticker(stock)
            test_data[stock] = ticker.history(start="2021-01-01", end="2022-12-31")['Close']
        except Exception as e:
            print(f"Error fetching test data for {stock}: {str(e)}")

    return pd.DataFrame(test_data)


# create predictive stock prices using Random Forest Regressor
def predictive_algorithm(stock_data):
    # Create an empty DataFrame to store the forecasted prices
    forecast = pd.DataFrame()

    for stock, data in stock_data.items():
        if "Close" in data:
            # Train the model on the historical data
            model = RandomForestRegressor()
            model.fit(data["Close"].to_numpy(), data["Close"].to_numpy())

            # Forecast the stock prices for 2021 and 2022
            forecast[stock] = model.predict(data["Close"].to_numpy())
        else:
            print(f"Skipping {stock} as 'Close' data is not available.")

    return forecast


# compare the forecasted amounts to the actuals we downloaded down. It uses RMSE, MSE and MAE to determine accuracy
def compare_forecast_to_actuals():
    # download the historical data for the 43 stocks
    stock_data = download_data()

    # forecast the stock prices for 2021 and 2022
    forecast = predictive_algorithm(stock_data)

    # get the actual stock prices for 2021 and 2022
    test_data = test_data()

    # compare the forecast to the actuals
    for stock in stocks:
        print(f"{stock}:")
        print(f"Forecast: {forecast[stock]}")
        print(f"Actual: {test_data[stock]}")

        # calculate the MSE, RMSE, and MAE
        mse = calculate_mse(forecast[stock], test_data[stock])
        rmse = calculate_rmse(forecast[stock], test_data[stock])
        mae = calculate_mae(forecast[stock], test_data[stock])

        # print the results
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")

compare_forecast_to_actuals()



if __name__ == "__main__":
    main()
'''
