import xgboost
import pandas as pd
import numpy as np

import sys

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

print("XGBoost version:", xgboost.__version__)

data_folder = "../../data/"

if len(sys.argv) < 7:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<author-embeddings> <n_estimators> <max_depth> <learning-rate> <subsample> <k-fold>")
    exit()

data_csv = data_folder + sys.argv[1]
n_estimators = int(sys.argv[2])
max_depth = int(sys.argv[3])
eta = float(sys.argv[4])
subsample = float(sys.argv[5])
k_fold = int(sys.argv[6])

data_df = pd.read_csv(data_csv, header=None, index_col=0)
h_index_df = pd.read_csv(data_folder + 'train.csv', index_col=0)

# h data
h_index = h_index_df.values[:,0]
N = h_index_df.shape[0]

print(N)

train_authors = h_index_df.index.values
train_h = h_index_df.loc[train_authors].values[:,0]
train_data = data_df.loc[train_authors].values

model = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, eta=eta, subsample=subsample, colsample_bytree=0.8)
# define model evaluation method
cv = RepeatedKFold(n_splits=k_fold, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, train_data, train_h, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MSE: %.3f %.3f (%.3f)' % (scores.mean(), scores.min(), scores.std()) )