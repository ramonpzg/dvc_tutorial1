import pandas as pd, numpy as np
import os, sys, yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]

split = params["split"]

# raw_data_path = os.path.join('data', 'raw', 'SeoulBikeData.csv')
raw_data_path = sys.argv[1]
train_path = os.path.join('data', 'processed', 'train.csv')
test_path = os.path.join('data', 'processed', 'test.csv')

data = pd.read_csv(raw_data_path, encoding='iso-8859-1')


# add date vars
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(['Date', 'Hour'], inplace=True)

data["Year"] = data['Date'].dt.year
data["Month"] = data['Date'].dt.month
data["Week"] = data['Date'].dt.isocalendar().week
data["Day"] = data['Date'].dt.day
data["Dayofweek"] = data['Date'].dt.dayofweek
data["Dayofyear"] = data['Date'].dt.dayofyear
data["Is_month_end"] = data['Date'].dt.is_month_end
data["Is_month_start"] = data['Date'].dt.is_month_start
data["Is_quarter_end"] = data['Date'].dt.is_quarter_end
data["Is_quarter_start"] = data['Date'].dt.is_quarter_start
data["Is_year_end"] = data['Date'].dt.is_year_end
data["Is_year_start"] = data['Date'].dt.is_year_start
data.drop('Date', axis=1, inplace=True)


data['Seasons_Autumn'] = np.where(data['Seasons'] == 'Autumn', 1, 0)
data['Seasons_Winter'] = np.where(data['Seasons'] == 'Winter', 1, 0)
data['Seasons_Summer'] = np.where(data['Seasons'] == 'Summer', 1, 0)
data['Seasons_Spring'] = np.where(data['Seasons'] == 'Spring', 1, 0)
data.drop('Seasons', axis=1, inplace=True)


data['Holiday_Yes'] = np.where(data['Holiday'] == 'Holiday', 1, 0)
data['Holiday_No'] = np.where(data['Holiday'] == 'No Holiday', 1, 0)
data.drop('Holiday', axis=1, inplace=True)


data['Functioning_Day_No'] = np.where(data['Functioning Day'] == 'No', 1, 0)
data['Functioning_Day_Yes'] = np.where(data['Functioning Day'] == 'Yes', 1, 0)
data.drop('Functioning Day', axis=1, inplace=True)

# Normalize columns
data.columns = ['rented_bike_count', 'hour', 'temperature', 'humidity', 'wind_speed', 'visibility', 
                'dew_point_temperature', 'solar_radiation', 'rainfall', 'snowfall', 'year', 
                'month', 'week', 'day', 'dayofweek', 'dayofyear', 'is_month_end', 'is_month_start',
                'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start',
                'seasons_autumn', 'seasons_winter', 'seasons_summer', 'seasons_spring',
                'holiday_yes', 'holiday_no', 'functioning_day_no', 'functioning_day_yes']



n_train = int(len(data) - len(data) * split)

df_train = data[:n_train].reset_index(drop=True)
df_test = data[n_train:].reset_index(drop=True)

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)