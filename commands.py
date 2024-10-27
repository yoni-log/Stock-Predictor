import tkinter as tk
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import customtkinter as ctk
import datetime as dt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models import Model


# Handle the submission of user inputs and update the UI with predictions
def submit(summary_text, verdict_label, accuracy_label, mse_label, preselected_stock, stock_period,
           model_combobox, plot_frame, interval_combobox):
    # Retrieve user inputs from the GUI elements
    ticker = preselected_stock.get()
    period = stock_period.get()
    model = model_combobox.get()
    interval = interval_combobox.get()

    # Retrieve stock data based on the selected ticker and date range
    stock_data = retrieve_stock(ticker, period, interval)

    # Generate stock price predictions using the selected model
    predictions, mse, loss, stock_data_with_predictions = predict_stock(model, stock_data, interval)

    # Calculate prediction accuracy
    accuracy = get_accuracy(predictions, stock_data["Close"].values)

    # Display the stock data and predictions on a plot
    display_plot(plot_frame, stock_data, stock_data_with_predictions)

    # Update accuracy and MSE labels with the calculated values
    accuracy_label.configure(text=f"Mean Accuracy: {accuracy:.2f}%")
    accuracy_label.grid()

    # Determine and display the verdict based on accuracy and MSE
    verdict = get_verdict(accuracy, mse, stock_data)
    verdict_label.configure(text=f"Verdict: " + verdict)
    verdict_label.grid()

    # Display the Mean Squared Error
    mse_label.configure(text=f"MSE: {str(float(mse))}")
    mse_label.grid()

    end_date = (dt.datetime.today() - dt.timedelta(days=int(period[0]))).date().strftime('%Y-%m-%d')
    start_date = dt.date.today().strftime('%Y-%m-%d')

    # Prepare a summary of the predictions and update the summary text area
    summary = (
        f"Stock Ticker: {ticker}\n"
        f"Date Range: {start_date} to {end_date}\n"
        f"Prediction Model: {model}\n"
        f"Interval: {interval}\n"
        f"Predicted Price: {float(predictions[-1]):.4f}\n"
        f"Mean Accuracy: {accuracy:.2f}%\n"
        f"Error: {mse:.2f}\n"
        f"Verdict: {verdict}"
    )
    summary_text.configure(state="normal")
    summary_text.delete("1.0", tk.END)
    summary_text.insert(tk.END, summary)
    summary_text.configure(state="disabled")
    summary_text.grid()


# Function to calculate the accuracy of the stock predictions
def get_accuracy(predicted, closing):
    maes = []
    for i in range(min(len(closing), len(predicted))):
        mae = np.mean(np.abs(closing[i] - predicted[i]))  # Calculate Mean Absolute Error for each prediction
        maes.append(mae)
    absolute_error = np.mean(maes)  # Calculate the average error across all predictions

    # Determine accuracy based on the difference between mean predicted and actual values
    if np.mean(predicted) - np.mean(closing) > 0:
        return abs(np.mean(predicted) - np.mean(closing) - 100)
    elif np.mean(predicted) - np.mean(closing) < 0:
        return abs(np.mean(predicted) - np.mean(closing) + 100)
    return None


# Function to determine the verdict based on accuracy, MSE, and stock data
def get_verdict(accuracy, mse, stock_data):
    # Extract current day's stock data
    current_close = stock_data["Close"].iloc[-1]
    current_open = stock_data["Open"].iloc[-1]
    current_high = stock_data["High"].iloc[-1]
    current_low = stock_data["Low"].iloc[-1]

    # Calculate daily average price
    daily_average = (current_close + current_open + current_high + current_low) / 4

    # Calculate the ratio of MSE to the daily average price
    mse_to_average_ratio = mse / daily_average

    # Determine the verdict based on accuracy and MSE-to-range ratio
    if accuracy > 65 and mse_to_average_ratio < 0.10:
        return "Great"
    elif accuracy > 60 and mse_to_average_ratio < 0.15:
        return "Good"
    elif accuracy > 55 and mse_to_average_ratio < 0.20:
        return "Fair"
    elif accuracy > 50 and mse_to_average_ratio < 0.25:
        return "Consider with Caution"
    else:
        return "Not Accurate"


# Function to retrieve stock data using yfinance
def retrieve_stock(ticker, period, interval):
    stock_data = yf.download(ticker, period=period, interval=interval)
    stock_data.reset_index(inplace=True)  # Reset index to include date as a column
    return stock_data


# Function to predict stock prices using the selected model
def predict_stock(model, stock_data, interval):
    stock_data = stock_data[["Open", "Volume", "Close"]]  # Select relevant columns for prediction
    model = Model(model, stock_data, extra_values_to_predict=20)  # Initialize the prediction model
    model.predict()  # Run the prediction

    stock_data_with_predictions = model.get_stock_data_with_predictions()  # Get stock data with appended predictions
    only_predictions = model.get_only_predictions()  # Get only the predicted values
    mse = model.get_mse()  # Get Mean Squared Error of the predictions
    loss = model.get_loss()  # Get the loss metric
    return only_predictions, mse, loss, stock_data_with_predictions


# Function to set up a combobox with predefined values in the tkinter UI
def set_combobox(frame, text, values, row, column, width=10, state="readonly", padx=5, pady=5, sticky="w"):
    label = ctk.CTkLabel(frame, text=text)  # Create label for the combobox
    label.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)
    combobox = ctk.CTkComboBox(frame, values=[str(v) for v in values], width=width, state=state)  # Create the combobox
    combobox.grid(row=row, column=column + 1, padx=padx, pady=pady, sticky=sticky)
    return combobox


# Function to display the stock data and predictions plot in the UI
def display_plot(frame, stock_data, stock_data_with_predictions):
    for widget in frame.winfo_children():  # Clear any existing plot in the frame
        widget.destroy()

    fig, ax = plt.subplots(figsize=(12, 4))  # Create a new figure and axis
    fig.patch.set_facecolor("#090a13")  # Set background color for the figure
    ax.set_facecolor("#202123")  # Set background color for the axis
    ax.spines["bottom"].set_color("white")  # Set color for the axis spines
    ax.spines["top"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.tick_params(axis="x", colors="white")  # Set color for x-axis ticks
    ax.tick_params(axis="y", colors="white")  # Set color for y-axis ticks

    # Plot the original stock data
    ax.plot(stock_data.index, stock_data["Close"], label="Original", color="skyblue")

    # Plot future predictions
    future_predictions = stock_data_with_predictions["Open"].tail(11)
    future_index = np.arange(len(stock_data.index), len(stock_data.index) + len(future_predictions))
    ax.plot(future_index, future_predictions.values, label="Future Predictions", color="orange")

    # Set title and labels
    ax.set_title("Stock Prices and Predictions", color="white")
    ax.set_xlabel("Time Steps", color="white")
    ax.set_ylabel("Price", color="white")
    ax.legend(facecolor="#070f1c", edgecolor="white")  # Set legend properties

    # Embed the plot into the tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
