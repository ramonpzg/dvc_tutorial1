import os, pickle, sys, yaml
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

params = yaml.safe_load(open("params.yaml"))["train"]

input_data = sys.argv[1]
output = os.path.join('models', 'rf_model.pkl')
seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]
max_feats = params["max_feats"]


X_train = pd.read_csv(input_data)
# X_test = pd.read_csv(os.path.join('data', 'processed', 'test.csv'))
y_train = X_train.pop('rented_bike_count')
# y_test = X_test.pop('rented_bike_count')


rf = RandomForestRegressor(n_estimators=n_est, 
                           min_samples_split=min_split, 
                           n_jobs=2,
                           max_features=max_feats,
                           random_state=seed,
                           oob_score=True
)

rf.fit(X_train.values, y_train.values)


with open(output, "wb") as fd:
    pickle.dump(rf, fd)