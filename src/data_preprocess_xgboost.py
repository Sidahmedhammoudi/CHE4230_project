import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost.callback import TrainingCallback
import matplotlib.pyplot as plt
from src.data_preprocess import DataPreprocessing 

class Preprocess():
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.data_processor = DataPreprocessing()

    def load_and_preprocess_data(self):
        """Loads and preprocesses the dataset."""
        data = self.data_processor.load_data(self.file_path)
        X = data.drop(columns=['electricity_demand_values[kw]', 'heat_demand_values[kw]'])
        y_electricity = data['electricity_demand_values[kw]']
        y_heat = data['heat_demand_values[kw]']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into training and testing sets
        X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity = train_test_split(
            X_scaled, y_electricity, test_size=0.2, random_state=42
        )
        
        X_train_heat, X_test_heat, y_train_heat, y_test_heat = train_test_split(
            X_scaled, y_heat, test_size=0.2, random_state=42
        )
        return (
            X_train_electricity, X_test_electricity, y_train_electricity, y_test_electricity,
            X_train_heat, X_test_heat, y_train_heat, y_test_heat
        )
       