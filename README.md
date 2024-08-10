
# ShibaTrader: Stock Market Analysis and Prediction
ShibaTrader is a stock analysis and prediction tool that leverages machine learning models such as Linear Regression, LSTM (Long Short-Term Memory), and Random Forest. Read the disclaimer before using this application.

![image](https://github.com/user-attachments/assets/33d660e5-8cb6-4d37-9209-5e2798d6cfb1)

# Features
* Stock Selection: Input your own stock ticker or choose from a preselected list of popular stocks such as TSLA, AAPL, AMZN, and more.
* Date Range: Specify the start and end dates to retrieve historical stock data.
* Prediction Models: Select from three models: Linear Regression, LSTM, and Random Forest.
* Data Interval: Choose data intervals ranging from 1 minute to 1 day.
* Stock Data Visualization: View predicted stock prices and compare them against actual closing prices.
* Prediction Summary: Displays the overall accuracy, mean squared error (MSE), and a verdict on prediction accuracy.

Ensure you have the following installed before running the program:
* Python 3.8+
* Required Packages: `customtkinter yfinance matplotlib numpy pandas sklearn torch`

# Steps to Use the Application
* Select a Stock: Choose a stock ticker from the dropdown list.
* Set Date Range: Use the date pickers to set the start and end dates for historical data retrieval.
* Choose Prediction Model: Select one of the available prediction models.
* Select Data Interval: Choose the interval for the stock data (e.g., 1 day, 1 hour). Read the disclaimer in the app if using an interval <1 day.
* Submit: Click the "Submit" button to run the prediction model and visualize the results.
* Review the Output: The application will display the prediction summary and plot the results on a graph.

# Model Descriptions
Linear Regression
* This model predicts the closing prices based on the opening prices using a linear regression approach.

Random Forest
* Random Forest is an ensemble model that uses multiple decision trees to make predictions. It provides a robust prediction by averaging the results from multiple trees.

LSTM (Long Short-Term Memory)
* LSTM is a type of recurrent neural network (RNN) designed for sequence prediction. It excels at predicting time series data due to its ability to maintain a memory of previous data points.

# Summary displayed
* Stock Ticker: The stock symbol you selected.
* Date Range: The range of dates for which the stock data was retrieved.
* Prediction Model: The model used for the prediction.
* Interval: The data interval selected.
* Predicted Price: The predicted stock price for the next period.
* Mean Accuracy: The accuracy of the model's predictions compared to the real values.
* Mean Squared Error (MSE): The error between predicted and actual prices.
* Verdict: A qualitative assessment of the prediction accuracy.
