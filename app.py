import atexit
import customtkinter as ctk
import models as mdl
import datetime as dt
import calendar as cd
import stock as stk
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from commands import *
from tkinter import messagebox
from tkcalendar import DateEntry

# Set global appearance mode and theme
ctk.set_appearance_mode("dark") # or "light"
ctk.set_default_color_theme("blue")

# Initialize main window
root = ctk.CTk()
root.iconbitmap("shiba.ico")
root.title("ShibaTrader")
root.geometry("1280x720")

# Frame for main content
frame = ctk.CTkFrame(root)
frame.pack(anchor='w', pady=10, padx=10, fill=ctk.BOTH, expand=True)

# Define default tickers
preselected_stocks = ['TSLA', 'AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NFLX', 'FB', 'NVDA', 'DIS', 'BABA']

stock_label = ctk.CTkLabel(frame, text="Ticker")
stock_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)
preselected_stock = ctk.CTkComboBox(frame, values=preselected_stocks, width=100)
preselected_stock.grid(row=0, column=1, padx=5, pady=5, sticky='w')

lower_date_label = ctk.CTkLabel(frame, text="Start Date")
lower_date_label.grid(row=1, column=0, sticky='w')
lower_date = DateEntry(frame, date_pattern='yyyy-mm-dd')
lower_date.grid(row=1, column=1, padx=5, pady=5, sticky='w')

upper_date_label = ctk.CTkLabel(frame, text="End Date")
upper_date_label.grid(row=2, column=0, sticky='w')
upper_date = DateEntry(frame, date_pattern='yyyy-mm-dd')
upper_date.grid(row=2, column=1, padx=5, pady=5, sticky='w')

model_label = ctk.CTkLabel(frame, text="Model")
model_label.grid(row=3, column=0, sticky='w')
model_combobox = ctk.CTkComboBox(frame, values=["Linear Regression", "LSTM", "Random Forest"], width=100)
model_combobox.grid(row=3, column=1, padx=5, pady=5, sticky='w')

interval_label = ctk.CTkLabel(frame, text="Interval")
interval_label.grid(row=4, column=0, sticky='w')
interval_combobox = ctk.CTkComboBox(frame, values=["1m", "5m", "1h", "1d"], width=100)
interval_combobox.grid(row=4, column=1, padx=5, pady=5, sticky='w')

interval_desc_text = ("*DISCLAIMER* yfinance has limitations in keeping data at intervals below 1 day\n"
                      "1h within past 730 days; 5m within past 60 days; 1m within past 7 days")
interval_desc = ctk.CTkLabel(frame, text=interval_desc_text)
interval_desc.grid(row=5, column=0, columnspan=3, sticky='w')

summary_label = ctk.CTkLabel(frame, text="Summary")
summary_label.grid(row=0, column=2, padx=100, pady=5, sticky='w')

summary_text = ctk.CTkTextbox(frame, height=120, width=250)
summary_text.grid(row=1, column=2, rowspan=3, padx=20, pady=5, sticky='w')

accuracy_label = ctk.CTkLabel(frame, text="Overall Accuracy:")
accuracy_label.grid(row=1, column=3, padx=10, pady=5, sticky='w')

mse_label = ctk.CTkLabel(frame, text="Mean Squared Error (MSE):")
mse_label.grid(row=2, column=3, padx=10, pady=5, sticky='w')

verdict_label = ctk.CTkLabel(frame, text="Accuracy Verdict:")
verdict_label.grid(row=3, column=3, padx=10, pady=5, sticky='w')

plot_frame = ctk.CTkFrame(root)
plot_frame.pack(anchor='w', pady=10, padx=10, fill=ctk.BOTH, expand=True)

submit_button = ctk.CTkButton(root, text="Submit",
                              command=lambda: submit(summary_text, verdict_label, accuracy_label, mse_label,
                                                     preselected_stock, lower_date, upper_date, model_combobox,
                                                     plot_frame, interval_combobox))
submit_button.pack(anchor='w', pady=10, padx=10)

root.mainloop()
