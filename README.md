# Stock-Price-Prediction
# Overview

This project builds a deep learning model using Long Short-Term Memory (LSTM) networks to predict Google stock prices based on historical data. The model learns temporal dependencies in time-series data and forecasts future stock trends.

Dataset

Training Data: Google_Stock_Price_Train.csv

Test Data: Google_Stock_Price_Test.csv

The dataset contains daily Google stock prices, including Open, High, Low, and Close values.

In this project, the Open price was used for prediction.

Workflow

Data Preprocessing

Imported and scaled the data using MinMaxScaler.

Created a 60-day lookback window to build training sequences.

Model Architecture

Built a stacked LSTM network with 4 LSTM layers and Dropout regularization.

Final dense layer outputs the predicted stock price.

Training

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

Epochs: 100

Batch Size: 32

Evaluation & Visualization

Predicted vs. actual Google stock prices were visualized using Matplotlib.

Model performance was evaluated with:

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

R² Score

Results

The model was able to capture stock price trends and produce predictions that closely follow real values. While not intended for financial decision-making, the project demonstrates the potential of deep learning in time-series forecasting.

Requirements

Python 3.x

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

How to Run

Clone the repository

Install dependencies:

pip install -r requirements.txt


Run the script:

 Stock_price_prediction.py
