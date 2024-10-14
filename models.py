import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pandas.core.common import random_state
from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sympy.vector import Gradient


# Define LSTM model class for predictions
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Define the main Model class to manage multiple prediction models
class Model(object):
    def __init__(self, model, stock_data, extra_values_to_predict):
        self.model = model
        self.stock_data = stock_data
        self.extra_values_to_predict = extra_values_to_predict

        self.mse = None
        self.loss = None
        self.stock_data_with_predictions = None
        self.y_pred_full = None

    # Select and predict following values on the appropriate model
    def predict(self):
        if self.model == "LSTM":
            self.lstm(self.stock_data, self.extra_values_to_predict)
        elif self.model == "Random Forest":
            self.random_forest(self.stock_data, self.extra_values_to_predict)
        elif self.model == "Linear Regression":
            self.linear_regression(self.stock_data, self.extra_values_to_predict)
        elif self.model == "Gradient Boosting":
            self.gradient_boosting(self.stock_data, self.extra_values_to_predict)

    # Implement LSTM prediction
    def lstm(self, stock_data, extra_values_to_predict):
        # Predicting stocks will be slower if NVIDIA Graphics Processing Unit(s) cannot be found
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # Extract features from data set
        X = stock_data[["Open"]].values
        y = stock_data["Close"].values

        # Scale the features and target
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # Create sequences of data for LSTM input
        seq_length = 20
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, seq_length)

        # Initialize the LSTM model with defined parameters
        model = LSTM(input_size=X_seq.shape[2], hidden_size=80, num_layers=2, output_size=1, dropout_rate=0.3)
        model = model.to(device)
        X_seq, y_seq = X_seq.to(device), y_seq.to(device)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        epochs = 30

        # Train the LSTM model
        for epoch in range(epochs):
            model.train()
            outputs = model(X_seq)
            loss = criterion(outputs, y_seq.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0: # To be implemented into GUI
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Evaluate the model and get predictions
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            y_seq_pred = model(X_seq)
            y_seq_pred_np = y_seq_pred.cpu().numpy()

        # Inverse transform the predictions to original scale
        y_seq_pred_unscaled = scaler_y.inverse_transform(y_seq_pred_np)
        y_unscaled = scaler_y.inverse_transform(y_seq.cpu().numpy().reshape(-1, 1))

        # Calculate the Mean Squared Error
        mse = mean_squared_error(y_unscaled, y_seq_pred_unscaled)

        # Update stock data with the latest prediction
        stock_data.loc[stock_data.index[-1], "Close"] = y_seq_pred_unscaled[-1]

        # Recursively predict the closing value for the next time step(s)
        if extra_values_to_predict > 0:
            next_open = y_seq_pred_unscaled[-1]
            next_data = pd.DataFrame({"Open": [next_open], "Volume": [1], "Close": [next_open]})
            stock_data = pd.concat([stock_data, next_data], ignore_index=True)
            return self.lstm(stock_data, extra_values_to_predict - 1)

        # Store the results and predictions
        self.mse = mse
        self.loss = loss.item()
        self.y_pred_full = y_seq_pred_unscaled
        self.stock_data_with_predictions = stock_data

    # Helper method to create sequences of data for LSTM input
    @staticmethod
    def create_sequences(stock_data, target, seq_length):
        xs, ys = [], []
        for i in range(len(stock_data) - seq_length):
            x = stock_data[i:i + seq_length]
            y = target[i + seq_length]
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

    # Method to implement Random Forest model prediction
    def random_forest(self, stock_data, extra_values_to_predict):
        # Extract feature and target data from the stock dataset
        X = stock_data[["Open", "Volume"]].values
        y = stock_data["Close"].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Random Forest model
        model = RandomForestRegressor(n_estimators=200, min_samples_split=50, random_state=1)
        model.fit(X_train, y_train)

        # Predict the full dataset and test set
        y_pred_full = model.predict(X)
        y_pred_test = model.predict(X_test)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred_test)
        print(extra_values_to_predict)

        # Update stock data with the latest prediction
        stock_data.loc[stock_data.index[-1], "Close"] = y_pred_full[-1]

        # Recursively predict the closing value for the next time step(s)
        if extra_values_to_predict > 0:
            next_open = y_pred_full[-1]
            next_data = pd.DataFrame({"Open": [next_open], "Volume": [1], "Close": [next_open]})
            stock_data = pd.concat([stock_data, next_data], ignore_index=True)
            return self.random_forest(stock_data, extra_values_to_predict - 1)

        # Store the results and predictions
        self.mse = mse
        self.loss = 0
        self.stock_data_with_predictions = stock_data
        self.y_pred_full = y_pred_full

    # Method to implement Linear Regression model prediction
    def linear_regression(self, stock_data, extra_values_to_predict):
        # Extract feature and target data from the stock dataset
        X = stock_data[["Open", "Volume"]].values
        y = stock_data["Close"].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the full dataset and test set
        y_pred_full = model.predict(X)
        y_pred_test = model.predict(X_test)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred_test)
        print(extra_values_to_predict)

        # Update stock data with the latest prediction
        stock_data.loc[stock_data.index[-1], "Close"] = y_pred_full[-1]

        # Recursively predict the closing value for the next time step(s)
        if extra_values_to_predict > 0:
            next_open = y_pred_full[-1]
            next_data = pd.DataFrame({"Open": [next_open], "Volume": [1], "Close": [next_open]})
            stock_data = pd.concat([stock_data, next_data])
            return self.linear_regression(stock_data, extra_values_to_predict - 1)

        # Store the results and predictions
        self.mse = mse
        self.loss = 0
        self.stock_data_with_predictions = stock_data
        self.y_pred_full = y_pred_full

    def gradient_boosting(self, stock_data, extra_values_to_predict):
        # Select the feature and target
        X = stock_data["Open"].values.reshape(-1, 1)  # Reshape to (n_samples, 1)
        y = stock_data["Close"].values.reshape(-1, 1)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize the Gradient Boosting Regressor
        reg = GradientBoostingRegressor(random_state=0)

        # Fit the model to the training data
        reg.fit(X_train, y_train)

        # Predict on the test set
        y_pred_full = reg.predict(X_test)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_pred_full)

        stock_data.loc[stock_data.index[-1], "Close"] = y_pred_full[-1]

        """###if extra_values_to_predict > 0:
            next_open = y_pred_full[-1]
            next_data = pd.DataFrame({"Open": [next_open], "Volume": [1], "Close": [next_open]})
            stock_data = pd.concat(next_data)
            return self.gradient_boosting(stock_data, extra_values_to_predict - 1)
"""
        self.mse = mse
        self.loss = 0
        self.y_pred_full = y_pred_full
        self.stock_data_with_predictions = stock_data

    # Getter method for Mean Squared Error (MSE)
    def get_mse(self):
        return self.mse

    # Getter method for Loss
    def get_loss(self):
        return self.loss

    # Getter method for the stock data with predictions included
    def get_stock_data_with_predictions(self):
        return self.stock_data_with_predictions

    # Getter method for only the predicted values
    def get_only_predictions(self):
        return self.y_pred_full
