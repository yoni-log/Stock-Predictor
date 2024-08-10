import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def lstm(stock_data, extra_values_to_predict):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    X = stock_data[["Open"]].values
    y = stock_data["Close"].values

    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    seq_length = 20
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    print("X_seq shape:", X_seq.shape)  # Debugging line

    model = LSTM(input_size=X_seq.shape[2], hidden_size=80, num_layers=2, output_size=1, dropout_rate=0.3)
    model = model.to(device)
    X_seq, y_seq = X_seq.to(device), y_seq.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    epochs = 30

    for epoch in range(epochs):
        model.train()
        outputs = model(X_seq)
        loss = criterion(outputs, y_seq.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        model.eval()
        y_seq_pred = model(X_seq)
    y_seq_pred_np = y_seq_pred.cpu().numpy()

    y_seq_pred_unscaled = scaler_y.inverse_transform(y_seq_pred_np)
    y_unscaled = scaler_y.inverse_transform(y_seq.cpu().numpy().reshape(-1, 1))

    mse = mean_squared_error(y_unscaled, y_seq_pred_unscaled)
    print(extra_values_to_predict)
    stock_data.loc[stock_data.index[-1], "Close"] = y_seq_pred_unscaled[-1]
    if extra_values_to_predict > 0:
        next_open = y_seq_pred_unscaled[-1]
        next_data = pd.DataFrame({"Open": [next_open], "Volume": [1], "Close": [next_open]})
        stock_data = pd.concat([stock_data, next_data], ignore_index=True)
        return lstm(stock_data, extra_values_to_predict - 1)

    with torch.no_grad():
        last_seq = X_scaled[-seq_length:]
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()
        next_day_pred_scaled = model(last_seq)
        next_day_pred = scaler_y.inverse_transform(next_day_pred_scaled.cpu().numpy().reshape(-1, 1))
    print(next_day_pred)

    return y_seq_pred_unscaled, mse, loss.item(), stock_data


def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def random_forest(stock_data, extra_values_to_predict):
    X = stock_data[["Open", "Volume"]].values
    y = stock_data["Close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(X_train, y_train)

    y_pred_full = model.predict(X)
    y_pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    print(extra_values_to_predict)
    stock_data.loc[stock_data.index[-1], "Close"] = y_pred_full[-1]
    if extra_values_to_predict > 0:
        next_open = y_pred_full[-1]
        next_data = pd.DataFrame({"Open": [next_open], "Volume": [1], "Close": [next_open]})
        stock_data = pd.concat([stock_data, next_data], ignore_index=True)
        return random_forest(stock_data, extra_values_to_predict - 1)

    return y_pred_full, mse, 0, stock_data


def linear_regression(stock_data, extra_values_to_predict):
    X = stock_data[["Open"]].values
    y = stock_data["Close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_full = model.predict(X)
    y_pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    print(extra_values_to_predict)
    stock_data.loc[stock_data.index[-1], "Close"] = y_pred_full[-1]
    if extra_values_to_predict > 0:
        next_open = y_pred_full[-1]
        next_data = pd.DataFrame({"Open": [next_open], "Volume": [1], "Close": [next_open]})
        stock_data = pd.concat([stock_data, next_data])
        return linear_regression(stock_data, extra_values_to_predict - 1)
    print(stock_data)

    return y_pred_full, mse, 0, stock_data
