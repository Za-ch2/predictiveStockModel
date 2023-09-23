# Predictive Stock Price Model

This project is focused on building and evaluating a predictive stock price model using historical stock price data and a Random Forest Regressor. The model aims to forecast stock prices for a list of 43 selected stocks for the years 2021 and 2022.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Financial forecasting is a critical aspect of investment and portfolio management. This project leverages Python and various libraries to predict stock prices for a portfolio of 43 stocks. Here's a brief overview of the project components:

- **Data Retrieval**: Historical stock price data is fetched from Yahoo Finance for the selected stocks. The data spans from 2010 to the most recent available date, with a fallback to all available data if there is no data for 2021.

- **Data Preparation**: The historical data is cleaned and prepared for model training. Missing data and outliers are handled appropriately.

- **Model Training**: A Random Forest Regressor model is trained on the historical stock price data to learn patterns and relationships.

- **Forecasting**: The trained model is used to forecast stock prices for the year 2021 and 2022.

- **Evaluation**: The forecasted prices are compared to the actual stock prices for 2021 and 2022, and metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) are calculated to assess model accuracy.

## Getting Started

To get started
