import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import torch.nn.functional as F
from MLP import TunableParameters, MLP

if len(sys.argv) < 2:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<author-embeddings>")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

# read training data and partition it in train and validation files
df_train = pd.read_csv('../../data/train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

# load the pre-processed features    
df_features = pd.read_csv(sys.argv[1], header=None)
df_features.rename(columns={0 :'author_id'}, inplace=True)
df_features.set_index('author_id', inplace=True)
df_features.sort_index(inplace=True)
df_features.head()

class AuthorDataset(Dataset):
    # The mapping file maps an author to its h-index
    def __init__(self, mapping_file):
        self.author_map = pd.read_csv(mapping_file)

    def __len__(self):
        return len(self.author_map)

    def __getitem__(self, idx):
        # Get the author id and its h-index
        author_id = self.author_map.iloc[idx, 0]
        h_index = self.author_map.iloc[idx, 1].astype(np.float32)
        features = df_features.loc[author_id,:].to_numpy(dtype=np.float32)
        return features, h_index

n_epochs = 10
n_k_folds = 5

input_size = df_features.shape[1]
output_size = 1

# Set the seed for random shuffles
torch.manual_seed(7)

parameters = [TunableParameters(512, 256, True, 0.2, 0.01), TunableParameters(512, 256, False, 0.0, 0.01), TunableParameters(512, 256, True, 0.2, 0.001), TunableParameters(256, 128, True, 0.2, 0.01), TunableParameters(256, 128, False, 0.0, 0.01), TunableParameters(256, 128, True, 0.2, 0.001)]

train_dataset = AuthorDataset('../../data/train.csv')

k_fold = KFold(n_splits=n_k_folds, shuffle=True)

for param in parameters:
    param.results = {}
    for fold, (train_ids, validation_ids) in enumerate(k_fold.split(train_dataset)):

        print(f'Performing fold {fold}...')
        print('##########################################')

        train_subsampler = SubsetRandomSampler(train_ids)
        validation_subsampler = SubsetRandomSampler(validation_ids)

        train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_subsampler)
        validation_loader = DataLoader(train_dataset, batch_size=64, sampler=validation_subsampler)

        model = MLP(input_size, param.n_hidden1, param.n_hidden2, output_size, param.use_dropout, param.dropout_p)
        # reset weights
        for layer in model.children():
             if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(0, n_epochs):
            fold_loss = 0.0
            data_size = 0

            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data_size += len(data)

                optimizer.zero_grad()
                output = model(data).squeeze()

                # MSE loss is used in this case
                loss = F.mse_loss(output, target)
                loss.backward()

                optimizer.step()

                fold_loss += F.mse_loss(output, target, reduction='sum')

            fold_loss /= data_size
            print(f'Epoch {epoch} completed, MSE loss: {fold_loss}')
        print("Training complete, switching to evaluation...")

        eval_loss = 0.0
        model.eval()

        with torch.no_grad():
            data_size = 0
            for i, (data, target) in enumerate(validation_loader):
                data, target = data.to(device), target.to(device)
                data_size += len(data)
                output = model(data).squeeze()
                eval_loss += F.mse_loss(output, target, reduction="sum").item()  # sum up batch loss

            eval_loss /= data_size

            print(f'MSE loss on fold {fold}: {eval_loss}')
            param.results[fold] = eval_loss  

# Print final results
print(f'KFold results for k={n_k_folds}:')
print('######################################')
for param in parameters: 
    print("Results for the following parameters:")
    param.print_parameters()
    average = 0.0
    for k, v in param.results.items():
        print(f'Fold {k}: {v}')
        average += v
    average /= len(param.results)
    print(f'Average value: {average}')