from commands import *
from tkcalendar import DateEntry

# Set global appearance mode and theme
ctk.set_appearance_mode("dark")
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
preselected_stocks = ["TSLA", "AAPL", "AMZN", "GOOGL", "MSFT", "NFL", "META", "NVDA", "DIS", "BABA"]

# Display ticker label & dropbox
stock_label = ctk.CTkLabel(frame, text="Ticker")
stock_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)
preselected_stock = ctk.CTkComboBox(frame, values=preselected_stocks, width=100)
preselected_stock.grid(row=0, column=1, padx=5, pady=5, sticky='w')

# Display start date label and date entry calendar
lower_date_label = ctk.CTkLabel(frame, text="Start Date")
lower_date_label.grid(row=1, column=0, sticky='w')
lower_date = DateEntry(frame, date_pattern="yyyy-mm-dd")
lower_date.grid(row=1, column=1, padx=5, pady=5, sticky='w')

# Display end date label and date entry calendar
upper_date_label = ctk.CTkLabel(frame, text="End Date")
upper_date_label.grid(row=2, column=0, sticky='w')
upper_date = DateEntry(frame, date_pattern="yyyy-mm-dd")
upper_date.grid(row=2, column=1, padx=5, pady=5, sticky='w')

# Display model label & dropbox
model_label = ctk.CTkLabel(frame, text="Model")
model_label.grid(row=3, column=0, sticky='w')
model_combobox = ctk.CTkComboBox(frame, values=["Linear Regression", "LSTM", "Random Forest", "Gradient Boosting"], width=100)
model_combobox.grid(row=3, column=1, padx=5, pady=5, sticky='w')

# Display interval label & dropbox
interval_label = ctk.CTkLabel(frame, text="Interval")
interval_label.grid(row=4, column=0, sticky='w')
interval_combobox = ctk.CTkComboBox(frame, values=["1m", "5m", "1h", "1d"], width=100)
interval_combobox.grid(row=4, column=1, padx=5, pady=5, sticky='w')

# Display disclaimer
interval_desc_text = ("*DISCLAIMER* yfinance has limitations in keeping data at intervals below 1 day\n"
                      "1h within past 730 days; 5m within past 60 days; 1m within past 7 days")
interval_desc = ctk.CTkLabel(frame, text=interval_desc_text)
interval_desc.grid(row=5, column=0, columnspan=3, sticky='w')

# Show summary box
summary_label = ctk.CTkLabel(frame, text="Summary")
summary_label.grid(row=0, column=2, padx=100, pady=5, sticky='w')
summary_text = ctk.CTkTextbox(frame, height=120, width=250)
summary_text.grid(row=1, column=2, rowspan=3, padx=20, pady=5, sticky='w')

# Display accuracy, error, and accuracy labels
accuracy_label = ctk.CTkLabel(frame, text="Overall Accuracy:")
accuracy_label.grid(row=1, column=3, padx=10, pady=5, sticky='w')

mse_label = ctk.CTkLabel(frame, text="Mean Squared Error (MSE):")
mse_label.grid(row=2, column=3, padx=10, pady=5, sticky='w')

verdict_label = ctk.CTkLabel(frame, text="Accuracy Verdict:")
verdict_label.grid(row=3, column=3, padx=10, pady=5, sticky='w')

# Get frame to visualize stocks' real and predicted valuations
plot_frame = ctk.CTkFrame(root)
plot_frame.pack(anchor='w', pady=10, padx=10, fill=ctk.BOTH, expand=True)

# Trains the model with the given data and displays data through unfilled labels and summary box
submit_button = ctk.CTkButton(root, text="Submit",
                              command=lambda: submit(summary_text, verdict_label, accuracy_label, mse_label,
                                                     preselected_stock, lower_date, upper_date, model_combobox,
                                                     plot_frame, interval_combobox))
submit_button.pack(anchor='w', pady=10, padx=10)

root.mainloop()
