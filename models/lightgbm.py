import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from src.data_preprocess_xgboost import Preprocess
import os

class Lightgbm_model():
    def __init__(self, data):
        self.data = data
        self.preprocess = Preprocess(data)
        # self.X_train_electricity, self.X_test_electricity, self.y_train_electricity, self.y_test_electricity,self.X_train_heat, self.X_test_heat, self.y_train_heat, self.y_test_heat = self.preprocess.load_and_preprocess_data()  # ← fix here
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = None  # ← store trained model here

    def train_model(self):
        self.model=    LGBMRegressor(
        n_estimators=1500,  # Increased boosting rounds
        learning_rate=0.05,  # Lower learning rate for better convergence
        num_leaves=30,  # Increased to allow more complex trees
        max_depth=6,  # Limit tree depth to prevent overfitting
        min_child_samples=50,  # Helps with regularization
        subsample=0.6,  # Use only 80% of data per tree (reduces overfitting)
        colsample_bytree=0.6,  # Use 80% of features per tree (improves generalization)
        reg_alpha=0.5,  # L1 regularization (reduces overfitting)
        reg_lambda=0.7,  # L2 regularization (reduces overfitting)
        random_state=42
    )

        return self.model
    


    def evaluate_lightgbm(self,model, X_train, y_train, X_test, y_test):
        
        """
        Evaluates a LightGBM model on training and testing data.

        Args:
            model: Trained LightGBM model.
            X_train (ndarray): Training input features.
            y_train (ndarray): Training targets.
            X_test (ndarray): Testing input features.
            y_test (ndarray): Testing targets.
            label (str): Optional label (e.g., 'electricity', 'heat') for display.

        Returns:
            dict: Dictionary with R² and MSE for train and test.
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # if label:
        #     label = f"[{label}] "

        # print(f"{label}Train R²: {train_r2:.4f}, Train MSE: {train_mse:.4f}")
        # print(f"{label}Test  R²: {test_r2:.4f}, Test  MSE: {test_mse:.4f}")

        return {
            "train_r2": train_r2,
            "train_mse": train_mse,
            "test_r2": test_r2,
            "test_mse": test_mse,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred
        }
    

    def plot_electricity(self, y_train,y_train_pred,y_test, y_test_pred,r2_train,r2_test,save_path):
        
        
        # Generate a combined scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(y_train, y_train_pred, alpha=0.6, label='Training data')
        plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test data')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Training and Test data for Xgboost Electricity Demand')
        plt.legend(loc='upper left')
        plt.plot([min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
                 [min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
                 color='red', lw=2, linestyle='--')
         
        # Add R² values as text in the plot
        plt.text(0.05, 0.87, f'Train R² = {r2_train:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.05, 0.82, f'Test R² = {r2_test:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.grid(True)
         
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_scatter_plot2_electricity_xgboost.png'))
        plt.show()

    def plot_heat(self, y_train,y_train_pred,y_test, y_test_pred,r2_train,r2_test,save_path):
        
        # Generate a combined scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(y_train, y_train_pred, alpha=0.6, label='Training data')
        plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test data')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Training and Test data for Xgboost Heat Demand')
        plt.legend(loc='upper left')
        plt.plot([min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
                 [min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
                 color='red', lw=2, linestyle='--')
         
        # Add R² values as text in the plot
        plt.text(0.05, 0.87, f'Train R² = {r2_train:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.05, 0.82, f'Test R² = {r2_test:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.grid(True)
         
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_scatter_plot2_heat_xgboost.png'))
        plt.show()
