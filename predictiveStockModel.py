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

'''def calculate_return(data):
    # NORMALIZATION - to measure all variables in comparable metric
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def show_statistics(returns):
    # instead of daily metrics we are after annual metrics
    # mean of annual return
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)


def show_mean_variance(returns, weights):
    # we are after the annual return
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()
                                                            * NUM_TRADING_DAYS, weights)))
    print("Expected portfolio mean (return): ", portfolio_return)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)


def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def generate_portfolios(returns, total_investment, allow_partial_purchase=False,
                       min_stock_weight=0.05, max_stock_weight=0.95):

    # randomly select a random number of stocks
    n_stocks = np.random.randint(1, len(stocks))

    # get the list of stocks to include in the portfolio
    stocks_to_include = stocks[np.random.choice(len(stocks), n_stocks, replace=False)]

    #setting budget
    total_investment = 10000

    # create the constraints
    constraints = []
    for i in range(n_stocks):
        constraints.append(
            {"type": "eq", "fun": lambda w: w[i] - min_stock_weight})
        constraints.append(
            {"type": "eq", "fun": lambda w: w[i] - max_stock_weight})

    # add the budget constraint
    constraints.append(
        {"type": "eq", "fun": lambda w: np.sum(w * stocks_to_include.price) - total_investment})

    # add the constraint to only allow stocks from the original list
    original_stocks = ["AAPL", "GOOG", "AMZN", "MSFT", "FB"]
    constraints.append(
        {"type": "ineq", "fun": lambda w: np.isin(w, original_stocks)})

    # generate the portfolios
    portfolio_means, portfolio_risks, portfolio_weights = generate_portfolios(
        returns[stocks_to_include], total_investment, allow_partial_purchase,
        constraints)

    return portfolio_means, portfolio_risks, portfolio_weights

def generate_additional_investment(returns, total_investment,
                                    allow_partial_purchase=False,
                                    min_stock_weight=0.05, max_stock_weight=0.95):

    # load the stock ticker CSV
    tickers_df = pd.read_csv(
        "C:\\Users\\PN174MM\\OneDrive - EY\\Desktop\\Personal Projects\\MarkModel\\stockdata\\USStockTicker.csv")

    # randomly select a random number of stocks
    n_stocks = np.random.randint(1, len(tickers_df))

    # get the list of stocks to include in the portfolio
    stocks_to_include = tickers_df["Ticker"].to_list()[:n_stocks]

    # generate the portfolios
    portfolio_means, portfolio_risks, portfolio_weights = generate_portfolios(
        returns[stocks_to_include], total_investment, allow_partial_purchase,
        constraints)

    return portfolio_means, portfolio_risks, portfolio_weights


def main():
    # get the stocks from the user
    print("Enter the stocks: ")
    stocks = []
    for stock in input().split():
        stocks.append(stock)

    # download the data
    data = download_data()

    # calculate the returns
    returns = calculate_return(data)

    # show the statistics of the returns
    show_statistics(returns)

    # generate portfolios for the base investment
    portfolio_means, portfolio_risks, portfolio_weights = generate_portfolios(
        returns, total_investment, allow_partial_purchase)

    # show the portfolios
    show_portfolios(portfolio_means, portfolio_risks)

    # generate portfolios for the additional investment
    additional_portfolio_means, additional_portfolio_risks, additional_portfolio_weights = generate_additional_investment(
        returns, 5000, allow_partial_purchase)

    # show the portfolios
    show_portfolios(additional_portfolio_means, additional_portfolio_risks)


if __name__ == "__main__":
    main()
'''