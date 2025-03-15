import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost.callback import TrainingCallback
import matplotlib.pyplot as plt
from src.data_preprocess_xgboost import Preprocess
import os


class R2MSECallback(TrainingCallback):
    def __init__(self, period=1, X_train=None, X_test=None, y_train=None, y_test=None):
        super().__init__()
        self.period = period
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            # Predict on training and testing data
            y_train_pred = model.predict(xgb.DMatrix(self.X_train))
            y_test_pred = model.predict(xgb.DMatrix(self.X_test))

            # Calculate R² and MSE for training data
            train_r2 = r2_score(self.y_train, y_train_pred)
            train_mse = mean_squared_error(self.y_train, y_train_pred)

            # Calculate R² and MSE for testing data
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)

            # Print the metrics
            print(f"Iteration {epoch}:")
            print(f"  Train R²: {train_r2:.4f}, Train MSE: {train_mse:.4f}")
            print(f"  Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")

        # Return False to stop training if needed
        return False


class XGBoostModel():
    def __init__(self, data):
        self.data = data
        self.preprocess = Preprocess(data)
        self.X_train_electricity, self.X_test_electricity, self.y_train_electricity, self.y_test_electricity,self.X_train_heat, self.X_test_heat, self.y_train_heat, self.y_test_heat = self.preprocess.load_and_preprocess_data()  # ← fix here
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = None  # ← store trained model here

    def train_model_electricity(self):
        dtrain_electricity = xgb.DMatrix(self.X_train_electricity, label=self.y_train_electricity)
        dtest_electricity = xgb.DMatrix(self.X_test_electricity, label=self.y_test_electricity)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse',
            'seed': 42
        }

        self.model = xgb.train(
            params,
            dtrain_electricity,
            num_boost_round=100,
            evals=[(dtrain_electricity, 'train'), (dtest_electricity, 'test')],
            callbacks=[R2MSECallback(period=10, X_train=self.X_train_electricity, X_test=self.X_test_electricity, y_train=self.y_train_electricity, y_test=self.y_test_electricity)],
            verbose_eval=False
        )

        return self.model

    def predict_electricity(self, X=None):
        # Use test set if X not provided
        if X is None:
            X_test = self.X_test_electricity
            X_train = self.X_train_electricity

        dtest = xgb.DMatrix(X_test)
        dtrain = xgb.DMatrix(X_train)
        y_test_pred = self.model.predict(dtest)
        y_train_pred = self.model.predict(dtrain)
        return y_test_pred, y_train_pred
    

    def train_model_heat(self):
        dtrain_heat = xgb.DMatrix(self.X_train_heat, label=self.y_train_heat)
        dtest_heat = xgb.DMatrix(self.X_test_heat, label=self.y_test_heat)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse',
            'seed': 42
        }

        self.model = xgb.train(
            params,
            dtrain_heat,
            num_boost_round=100,
            evals=[(dtrain_heat, 'train'), (dtest_heat, 'test')],
            callbacks=[R2MSECallback(period=10, X_train=self.X_train_heat, X_test=self.X_test_heat, y_train=self.y_train_heat, y_test=self.y_test_heat)],
            verbose_eval=False
        )

        return self.model

    def predict_heat(self, X=None):
        # Use test set if X not provided
        if X is None:
            X_test = self.X_test_heat
            X_train = self.X_train_heat

        dtest = xgb.DMatrix(X_test)
        dtrain = xgb.DMatrix(X_train)
        y_test_pred = self.model.predict(dtest)
        y_train_pred = self.model.predict(dtrain)
        return y_test_pred, y_train_pred
    
    def evaluate(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return r2, mse

    def plot_electricity(self, y_train_pred, y_test_pred,r2_train,r2_test,save_path):
        y_train_true_electricity = self.y_train_electricity
        y_test_true_electricity = self.y_test_electricity
        
        # Generate a combined scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(y_train_true_electricity, y_train_pred, alpha=0.6, label='Training data')
        plt.scatter(y_test_true_electricity, y_test_pred, alpha=0.6, label='Test data')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Training and Test data for Xgboost Electricity Demand')
        plt.legend(loc='upper left')
        plt.plot([min(np.concatenate((y_train_true_electricity, y_test_true_electricity))), max(np.concatenate((y_train_true_electricity, y_test_true_electricity)))],
                 [min(np.concatenate((y_train_true_electricity, y_test_true_electricity))), max(np.concatenate((y_train_true_electricity, y_test_true_electricity)))],
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

    def plot_heat(self, y_train_pred, y_test_pred,r2_train,r2_test,save_path):
        y_train_true_heat = self.y_train_heat
        y_test_true_heat = self.y_test_heat
        
        # Generate a combined scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(y_train_true_heat, y_train_pred, alpha=0.6, label='Training data')
        plt.scatter(y_test_true_heat, y_test_pred, alpha=0.6, label='Test data')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Training and Test data for Xgboost Heat Demand')
        plt.legend(loc='upper left')
        plt.plot([min(np.concatenate((y_train_true_heat, y_test_true_heat))), max(np.concatenate((y_train_true_heat, y_test_true_heat)))],
                 [min(np.concatenate((y_train_true_heat, y_test_true_heat))), max(np.concatenate((y_train_true_heat, y_test_true_heat)))],
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

