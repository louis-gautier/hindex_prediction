import xgboost
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from random import shuffle

print("XGBoost version:", xgboost.__version__)

data_folder = "../../data/"
authors_csv = data_folder + "allenai-specter-pca-256-deepwalk-128.csv"
sentence_csv = data_folder + "transformer-embeddings-all-mpnet-base-v2-pca-256-post.csv"

data_df = pd.read_csv(authors_csv, header=None, index_col=0)
sentence_df = pd.read_csv(sentence_csv, header=None, index_col=0)
h_index_df = pd.read_csv(data_folder + 'train.csv', index_col=0)
test_df = pd.read_csv(data_folder + 'test.csv', index_col=0)

authors = h_index_df.index.values
#print(sentence_df)
#print(authors)
# print(authors_df.loc[authors])
# print(sentence_df.loc[authors])

#data_df = pd.concat([authors_df.loc[authors], sentence_df.loc[authors]], axis=1)

#print(data_df)

# h data
h_index = h_index_df.values[:,0]
N = h_index_df.shape[0]

print(N)

# index selection
indices = [i for i in range(N)]
shuffle(indices)

N_test = 35000
train_indicies = indices[N_test:]
test_indicies = indices[:N_test]

# test_authors = test_df['author'].values
train_authors = h_index_df.index.values[train_indicies]
test_authors = h_index_df.index.values[test_indicies]

train_h = h_index_df.loc[train_authors].values[:,0]
test_h = h_index_df.loc[test_authors].values[:,0]

train_data = data_df.loc[train_authors].values
test_data = data_df.loc[test_authors].values

# print(train_data)
# print(train_h)

model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model.fit(train_data, train_h)

prediction = model.predict(test_data)
# print(prediction, test_h)
print("MSE: ", np.sum((prediction - test_h)**2)/N_test)