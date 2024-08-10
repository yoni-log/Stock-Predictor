import tkinter as tk
import models as mdl
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import customtkinter as ctk
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from tkinter import messagebox


def set_lower_combobox(parent):
    lower_month = ttk.Combobox(parent, values=list(range(1, 13)), state="readonly", width=5, style='Custom.TCombobox')
    lower_month.grid(row=0, column=0, padx=5)

    lower_day = ttk.Combobox(parent, values=list(range(1, 32)), state="readonly", width=5, style='Custom.TCombobox')
    lower_day.grid(row=0, column=1, padx=5)

    lower_year = ttk.Combobox(parent, values=list(range(1950, dt.date.today().year + 1)), state="readonly", width=7, style='Custom.TCombobox')
    lower_year.grid(row=0, column=2, padx=5)

    return lower_month, lower_day, lower_year


def set_upper_combobox(parent):
    upper_month = ttk.Combobox(parent, values=list(range(1, 13)), state="readonly", width=5, style='Custom.TCombobox')
    upper_month.grid(row=0, column=0, padx=5)

    upper_day = ttk.Combobox(parent, values=list(range(1, 32)), state="readonly", width=5, style='Custom.TCombobox')
    upper_day.grid(row=0, column=1, padx=5)

    upper_year = ttk.Combobox(parent, values=list(range(2000, dt.date.today().year + 1)), state="readonly", width=7, style='Custom.TCombobox')
    upper_year.grid(row=0, column=2, padx=5)

    return upper_month, upper_day, upper_year


def submit(summary_text, verdict_label, accuracy_label, mse_label, preselected_stock, lower_date, upper_date, model_combobox, plot_frame, interval_combobox):
    ticker = preselected_stock.get()
    start_date = lower_date.get_date()
    end_date = upper_date.get_date()
    model = model_combobox.get()
    interval = interval_combobox.get()

    stock_data = retrieve_stock(ticker, start_date, end_date, interval=interval)
    predictions, mse, loss, stock_data_with_predictions = predict_stock(model, stock_data, interval)
    accuracy = get_accuracy(predictions, stock_data["Close"].values)

    display_plot(plot_frame, stock_data, stock_data_with_predictions)

    accuracy_label.configure(text=f"Mean Accuracy: {accuracy:.2f}%")
    accuracy_label.grid()

    verdict = get_verdict(accuracy, mse, stock_data)

    verdict_label.configure(text=f"Verdict: " + verdict)
    verdict_label.grid()

    mse_label.configure(text=f"MSE: {str(float(mse))}")
    mse_label.grid()

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
    summary_text.configure(state='normal')
    summary_text.delete('1.0', tk.END)
    summary_text.insert(tk.END, summary)
    summary_text.configure(state='disabled')
    summary_text.grid()


def get_accuracy(predicted, closing):
    print("pred len: " + str(len(predicted)) + "\n closing pred len: " + str(len(closing)))

    maes = []
    for i in range(min(len(closing), len(predicted))):
        mae = np.mean(np.abs(closing[i] - predicted[i]))
        maes.append(mae)
    absolute_error = np.mean(maes)
    print(f"Mean Absolute Error: {absolute_error}")

    if np.mean(predicted) - np.mean(closing) > 0:
        print(f"Mean accuracy: {abs(np.mean(predicted) - np.mean(closing) - 100)}")
        return abs(np.mean(predicted) - np.mean(closing) - 100)
    elif np.mean(predicted) - np.mean(closing) < 0:
        print(f"Mean accuracy: {abs(np.mean(predicted) - np.mean(closing) + 100)}")
        return abs(np.mean(predicted) - np.mean(closing) + 100)
    return None


def get_verdict(accuracy, mse, stock_data):
    # Extract current day's stock data
    current_close = stock_data["Close"].iloc[-1]
    current_open = stock_data["Open"].iloc[-1]
    current_high = stock_data["High"].iloc[-1]
    current_low = stock_data["Low"].iloc[-1]

    # Calculate daily range
    daily_average = (current_close + current_open + current_high + current_low) / 4

    # Calculate the ratio of MSE to the daily range
    mse_to_average_ratio = mse / daily_average
    # Define thresholds for the MSE-to-range ratio
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


def retrieve_stock(ticker, start_date, end_date, interval):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    stock_data.reset_index(inplace=True)
    return stock_data


def predict_stock(model, stock_data, interval):
    stock_data = stock_data[["Open", "Volume", "Close"]]

    models_dict = {
        "LSTM": mdl.lstm,
        "Random Forest": mdl.random_forest,
        "Linear Regression": mdl.linear_regression
    }
    data, mse, loss, stock_data_with_predictions = models_dict[model](stock_data, 10) # 10: values to predict (integer)

    return data, mse, loss, stock_data_with_predictions


def set_combobox(frame, text, values, row, column, width=10, state='readonly', padx=5, pady=5, sticky='w'):
    label = ctk.CTkLabel(frame, text=text)
    label.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)
    combobox = ctk.CTkComboBox(frame, values=[str(v) for v in values], width=width, state=state)
    combobox.grid(row=row, column=column + 1, padx=padx, pady=pady, sticky=sticky)
    return combobox


def display_plot(frame, stock_data, stock_data_with_predictions):
    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#090a13")
    ax.set_facecolor('#202123')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.plot(stock_data.index, stock_data['Close'], label='Original', color='skyblue')

    future_predictions = stock_data_with_predictions["Open"].tail(11)
    future_index = np.arange(len(stock_data.index), len(stock_data.index) + len(future_predictions))
    ax.plot(future_index, future_predictions.values, label='Future Predictions', color='orange')

    ax.set_title('Stock Prices and Predictions', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Price', color='white')
    ax.legend(facecolor='#070f1c', edgecolor='white')

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)