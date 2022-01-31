import os
import sys
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from MLP import TunableParameters, MLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

# read training data and partition it in train and validation files
df_train = pd.read_csv('../../data/train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

msk = np.random.rand(len(df_train)) < 0.8

internal_train = df_train[msk]
internal_train.to_csv('../../data/internal-train.csv')
internal_validation = df_train[~msk]
internal_validation.to_csv('../../data/internal-validation.csv')

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
        author_id = self.author_map.iloc[idx, 1]
        h_index = self.author_map.iloc[idx, 2].astype(np.float32)
        features = df_features.loc[author_id,:].to_numpy(dtype=np.float32)
        return features, h_index

train_losses = []
test_losses = []

def train(model, device, train_loader, optimizer, epoch):
    log_interval=1000
    model.train() #set model in train mode
    train_loss = 0.0
    data_size = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        
        # MSE loss is used in this case
        loss = F.mse_loss(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        train_loss += F.mse_loss(output, target, reduction="sum").item()
        data_size += len(data)
    train_loss /= data_size
    train_losses.append(train_loss)
    
            
def test(model, device, test_loader):
    model.eval() #set model in test mode
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += F.mse_loss(output, target, reduction="sum").item()  # sum up batch loss
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: MSE loss on test set: {:.4f}\n'.format(
        test_loss))

train_dataset = AuthorDataset('../../data/internal-train.csv')
validation_dataset = AuthorDataset('../../data/internal-validation.csv')

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
validation_loader = DataLoader(validation_dataset,batch_size=1000,shuffle=True)

input_size = df_features.shape[1]
if len(sys.argv) > 3:
    hidden = [int(sys.argv[2]), int(sys.argv[3])]
else:
    hidden = [64, 64]
output_size = 1


if len(sys.argv) > 4:
    n_epochs = int(sys.argv[4])
else:
    n_epochs = 1

if len(sys.argv) > 5:
    use_dropout = True
    dropout_p = float(sys.argv[5])
    if dropout_p == 0.0:
        use_dropout = False
else:
    use_dropout = False
    dropout_p = 0.1

if len(sys.argv) > 6:
    lr  = float(sys.argv[6])
else:
    lr = 0.01

model_params = TunableParameters(hidden[0], hidden[1], use_dropout, dropout_p, lr)
model = MLP(input_size,model_params.n_hidden1, model_params.n_hidden2, output_size, use_dropout=model_params.use_dropout, dropout_p=model_params.dropout_p)
model.to(device)
#sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam_optimizer = torch.optim.Adam(model.parameters(), lr=model_params.lr)

#print(model)
model_params.print_parameters()

print("STARTING TRAINING")
print("########################################")
for epoch in range(0, n_epochs):
    train(model, device, train_loader, adam_optimizer, epoch)
    test(model, device, validation_loader)

df_test = pd.read_csv('../../data/test.csv', index_col=0, dtype={'author': np.int64, 'hindex': np.float32}, delimiter=',')

model.eval()
for i, row in df_test.iterrows():
    author_id = row['author']
    features = df_features.loc[author_id,:].to_numpy(dtype=np.float32)
    features = torch.from_numpy(features).to(device)
    h_index = int(round(model(features).item()))
    df_test.at[i, 'hindex']  = h_index

df_test = df_test.astype({'hindex':np.int32})
test_output_path = sys.argv[7] + '-' + model_params.get_param_string() + '.csv'
df_test.to_csv(test_output_path, sep=',')
print("Saved test predictions to", test_output_path)

# Store training and validation losses in order to plot them later
if len(sys.argv) > 8:
    output_path = sys.argv[8] + '-' + model_params.get_param_string() + '.pkl'
    output_data = {"title": model_params.get_param_string(), "train_losses": train_losses, "test_losses": test_losses}
    pickle.dump(output_data, open(output_path, "wb" ))
    print("Saved train and validation losses to", output_path)
