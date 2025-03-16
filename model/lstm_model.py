import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from src.data_preprocess import DataPreprocessing 
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class LSTM_model:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.data_processor = DataPreprocessing()

    def load_and_preprocess_data(self):
        """Loads and preprocesses the dataset."""
        data = self.data_processor.load_data(self.file_path)
        

        # Split features and target
        X = data.drop(columns=['electricity_demand_values[kw]', 'heat_demand_values[kw]'])
        y_electricity = data['electricity_demand_values[kw]']
        y_heat = data['heat_demand_values[kw]']

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y_electricity, y_heat

    def reshape_data_for_lstm(self, X, y, time_steps=24):
        """Reshape data for LSTM (as described in the document)."""
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])
        return np.array(X_seq), np.array(y_seq)

    def split_data(self):
        # Load and preprocess data
        X_scaled, y_electricity, y_heat = self.load_and_preprocess_data()

        # Reshape data for electricity demand prediction
        X_seq_electricity, y_seq_electricity = self.reshape_data_for_lstm(X_scaled, y_electricity, time_steps=24)

        # Reshape data for heat demand prediction
        X_seq_heat, y_seq_heat = self.reshape_data_for_lstm(X_scaled, y_heat, time_steps=24)

        # Split into training and testing sets
        X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity = train_test_split(
            X_seq_electricity, y_seq_electricity, test_size=0.2, random_state=42
        )

        X_train_heat, X_test_heat, y_train_heat, y_test_heat = train_test_split(
            X_seq_heat, y_seq_heat, test_size=0.2, random_state=42
        )

        return (
            X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity,
            X_train_heat, X_test_heat, y_train_heat, y_test_heat
        )
       
    
    def compile_model_electricity(self,X_train_electricity):
        """Compiles the LSTM model."""
        # X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity, X_train_heat, X_test_heat, y_train_heat, y_test_heat = self.split_data()
        input_shape = (X_train_electricity.shape[1], X_train_electricity.shape[2])
        self.model = Sequential([
    LSTM(50, return_sequences=True, input_shape=input_shape),
    LSTM(50),
    Dense(25, activation='relu'),
    Dense(1)
])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model
    
    

    def predict_electricity(self, X_train_name, X_test_name, y_train_name, y_test_name):
        """Trains and predicts using the LSTM model on the selected dataset by variable name.
    
    Args:
        X_train_name (str): Variable name for training input features
        X_test_name (str): Variable name for test input features
        y_train_name (str): Variable name for training targets
        y_test_name (str): Variable name for test targets

    Returns:
        Tuple of (y_train_pred, y_test_pred)
    """
    # Retrieve all split data
        values = self.split_data()
        names = [
            "X_train_electricity", "X_test_electricity", "y_train_electricity", "y_test_electricity",
            "X_train_heat", "X_test_heat", "y_train_heat", "y_test_heat"
        ]
        split = dict(zip(names, values))
    
        # Fetch the specific inputs by name
        X_train = split.get(X_train_name)
        X_test = split.get(X_test_name)
        y_train = split.get(y_train_name)
        y_test = split.get(y_test_name)
    
        # Safety check
        # Safety check (FIXED)
        if any(v is None for v in [X_train, X_test, y_train, y_test]):
            raise ValueError("One or more dataset names are invalid or missing from split_data() output.")

    
        # Reshape for LSTM if needed (expects 3D input)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
        # Train the model
        self.train_model_electricity(X_train, y_train, X_test, y_test)
    
        # Predict using trained model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
    
        return y_train_pred, y_test_pred
    
    def predict_heat(self, X_train_name, X_test_name, y_train_name, y_test_name):
        """Trains and predicts using the LSTM model on the selected dataset by variable name.
    
    Args:
        X_train_name (str): Variable name for training input features
        X_test_name (str): Variable name for test input features
        y_train_name (str): Variable name for training targets
        y_test_name (str): Variable name for test targets

    Returns:
        Tuple of (y_train_pred, y_test_pred)
    """
    # Retrieve all split data
        values = self.split_data()
        names = [
            "X_train_electricity", "X_test_electricity", "y_train_electricity", "y_test_electricity",
            "X_train_heat", "X_test_heat", "y_train_heat", "y_test_heat"
        ]
        split = dict(zip(names, values))
    
        # Fetch the specific inputs by name
        X_train = split.get(X_train_name)
        X_test = split.get(X_test_name)
        y_train = split.get(y_train_name)
        y_test = split.get(y_test_name)
    
        # Safety check
        # Safety check (FIXED)
        if any(v is None for v in [X_train, X_test, y_train, y_test]):
            raise ValueError("One or more dataset names are invalid or missing from split_data() output.")

    
        # Reshape for LSTM if needed (expects 3D input)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
        # Train the model
        self.train_model_heat(X_train, y_train, X_test, y_test)
    
        # Predict using trained model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
    
        return y_train_pred, y_test_pred


    

    
    def train_model_electricity(self,X_train_electricity, y_train_electricity,X_test_electricity, y_test_electricity):
        model = self.compile_model_electricity(X_train_electricity)
        # X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity, X_train_heat, X_test_heat, y_train_heat, y_test_heat = self.split_data()
        electricity_model= model.fit(X_train_electricity, y_train_electricity, epochs=50, batch_size=32, validation_data=(X_test_electricity, y_test_electricity))
        return electricity_model
    
    def compile_model_heat(self,X_train_heat):
        """Compiles the LSTM model."""
        X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity, X_train_heat, X_test_heat, y_train_heat, y_test_heat = self.split_data()
        input_shape = (X_train_heat.shape[1], X_train_heat.shape[2])
        self.model = Sequential([
    LSTM(50, return_sequences=True, input_shape=input_shape),
    LSTM(50),
    Dense(25, activation='relu'),
    Dense(1)
])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model
        

    def train_model_heat(self,X_train_heat, y_train_heat,X_test_heat, y_test_heat):
        model = self.compile_model_heat(X_train_heat)
        X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity, X_train_heat, X_test_heat, y_train_heat, y_test_heat = self.split_data()
        heat_model= model.fit(X_train_heat, y_train_heat, epochs=50, batch_size=32, validation_data=(X_test_heat, y_test_heat))
        return heat_model
    
    def evaluate(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return r2, mse
    
    def plot_electricity(self, y_train_pred,y_train_electricity, y_test_pred,y_test_electricity,r2_train,r2_test,save_path):
        y_train_true_flat = y_train_electricity.flatten()
        y_train_pred_flat = y_train_pred.flatten()
        y_test_true_flat = y_test_electricity.flatten()
        y_test_pred_flat = y_test_pred.flatten()
        
        # Generate a combined scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(y_train_true_flat, y_train_pred_flat, alpha=0.6, label='Training data')
        plt.scatter(y_test_true_flat, y_test_pred_flat, alpha=0.6, label='Test data')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Training and Test data for Electricity Demand with LSTM model')
        plt.legend(loc='upper left')
        plt.plot([min(np.concatenate((y_train_true_flat, y_test_true_flat))), max(np.concatenate((y_train_true_flat, y_test_true_flat)))],
                 [min(np.concatenate((y_train_true_flat, y_test_true_flat))), max(np.concatenate((y_train_true_flat, y_test_true_flat)))],
                 color='red', lw=2, linestyle='--')
         
        # Add R² values as text in the plot
        plt.text(0.05, 0.87, f'Train R² = {r2_train:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.05, 0.82, f'Test R² = {r2_test:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.grid(True)
         
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_scatter_plot2_electricitylstm.png'))
        plt.show()

    def plot_heat(self,y_train_pred,y_train_heat, y_test_pred,y_test_heat,r2_train,r2_test,save_path):
        y_train_true_flat = y_train_heat.flatten()
        y_train_pred_flat = y_train_pred.flatten()
        y_test_true_flat = y_test_heat.flatten()
        y_test_pred_flat = y_test_pred.flatten()
        
        # Generate a combined scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(y_train_true_flat, y_train_pred_flat, alpha=0.6, label='Training data')
        plt.scatter(y_test_true_flat, y_test_pred_flat, alpha=0.6, label='Test data')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Training and Test data for LSTM Heat Demand')
        plt.legend(loc='upper left')
        plt.plot([min(np.concatenate((y_train_true_flat, y_test_true_flat))), max(np.concatenate((y_train_true_flat, y_test_true_flat)))],
                 [min(np.concatenate((y_train_true_flat, y_test_true_flat))), max(np.concatenate((y_train_true_flat, y_test_true_flat)))],
                 color='red', lw=2, linestyle='--')
         
        # Add R² values as text in the plot
        plt.text(0.05, 0.87, f'Train R² = {r2_train:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.05, 0.82, f'Test R² = {r2_test:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.grid(True)
         
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_scatter_plot2_heat_lstm.png'))
        plt.show()

