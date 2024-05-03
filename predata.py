import pandas as pd
import numpy as np

# Load the data from the uploaded files
data_500 = pd.read_csv('./OutcomeData/model2_output_500.txt', sep=" ", names=["Time", "Value"])
data_1000 = pd.read_csv('./OutcomeData/model2_output_1000.txt', sep=" ", names=["Time", "Value"])
data_2000 = pd.read_csv('./OutcomeData/model2_output_2000.txt', sep=" ", names=["Time", "Value"])

# Apply a moving average with a window of 3
data_500_smooth = data_500['Value'].rolling(window=3, min_periods=1, center=True).mean()
data_1000_smooth = data_1000['Value'].rolling(window=3, min_periods=1, center=True).mean()
data_2000_smooth = data_2000['Value'].rolling(window=3, min_periods=1, center=True).mean()

data_500['Smoothed'] = data_500_smooth
data_1000['Smoothed'] = data_1000_smooth
data_2000['Smoothed'] = data_2000_smooth

data_500, data_1000, data_2000
