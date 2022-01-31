import xgboost
import pandas as pd
import numpy as np

import sys

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from random import shuffle

print("XGBoost version:", xgboost.__version__)

data_folder = "../../data/"
submission_folder = "../submissions/"

if len(sys.argv) < 8:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<author-embeddings> <n_estimators> <max_depth> <learning-rate> <subsample> <k-fold> <submission_csv>")
    exit()

data_csv = data_folder + sys.argv[1]
out_csv = submission_folder + sys.argv[7]
n_estimators = int(sys.argv[2])
max_depth = int(sys.argv[3])
eta = float(sys.argv[4])
subsample = float(sys.argv[5])
k_fold = int(sys.argv[6])

data_df = pd.read_csv(data_csv, header=None, index_col=0)
h_index_df = pd.read_csv(data_folder + 'train.csv', index_col=0)
test_df = pd.read_csv(data_folder + 'test.csv', index_col=0)

# h data
h_index = h_index_df.values[:,0]
N = h_index_df.shape[0]

# indices selection
N_test = int(0.2 * N)
indices = [i for i in range(N)]
shuffle(indices)
train_indices = indices[N_test:]
validation_indices = indices[:N_test]

# authors selection
train_authors = h_index_df.index.values[train_indices]
validation_authors = h_index_df.index.values[validation_indices]
test_authors = test_df['author'].values

# h-index values
train_h = h_index[train_indices]
validation_h = h_index[validation_indices]

# authors data
train_data = data_df.loc[train_authors].values
validation_data = data_df.loc[validation_authors].values
test_data = data_df.loc[test_authors].values

# training
print("Number of entries for model training:", train_data.shape[0])
model = xgboost.XGBRegressor(n_estimators=1000, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model.fit(train_data, train_h)

# validation
validation = model.predict(validation_data)
print('MSE of validation set:', np.mean((validation - validation_h)**2))

# prediction
prediction = model.predict(test_data)
test_df['hindex'] = prediction

# export
test_df.to_csv(out_csv, index=False)