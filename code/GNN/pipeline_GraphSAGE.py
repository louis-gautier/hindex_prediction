import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphSAGE
from sklearn.preprocessing import StandardScaler
import sys

if len(sys.argv) < 7:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<author-full-embeddings> <model-size> <size-layer1> <size-layers2-3> <dropout-rate> <aggregation-function>")
    exit()

# Get the full author embeddings 
embedding_name = str(sys.argv[1])
nb_features = int(sys.argv[2])

# The following class represents our graph data
class AutorGraph():
  def __init__(self,features_file,edges_file,nb_features):
    # Getting the author embeddings
    data_df = pd.read_csv(features_file,header=None)
    data_df.rename(columns={0 :'author_id', (nb_features+1):'hindex'}, inplace=True)

    # Defining the features dataframe
    features_df = data_df.loc[:,np.logical_and(data_df.columns != 'author_id', data_df.columns !='hindex')]
    # We scale every feature, taking its z-score
    std_scaler = StandardScaler()
    features_scaled = std_scaler.fit_transform(features_df.to_numpy())
    features_df = pd.DataFrame(features_scaled)
    # Defining the labels dataframe
    labels_df = data_df[["hindex"]]
    # Defining the validation set: we assign a h-index of -2 for all authors in the validation set to build the validation_mask easily, and save their real h-index for later
    hindex_validation = []
    for i in range(len(labels_df.index)):
      if labels_df.loc[i,"hindex"]!=-1:
        if(np.random.rand()<0.2):
          hindex_validation.append(labels_df.loc[i,"hindex"])
          labels_df.at[i,"hindex"]=-2
    # Defining train, test and validation masks
    test_mask = labels_df.where(labels_df.hindex==-1, other=0).values.T[0]
    test_mask = test_mask.astype(bool)
    validation_mask = labels_df.where(labels_df.hindex==-1.01, other=0).values.T[0]
    validation_mask = validation_mask.astype(bool)
    train_mask = np.logical_not(np.logical_or(test_mask,validation_mask))
    # Defining the nodes tensor
    x = torch.tensor(features_df.values,dtype=torch.float)
    # Defining the labels tensor
    y = torch.tensor(labels_df.values,dtype=torch.float)
    # Building the edges tensor
    # Dictionary mapping author_id to indices in the torch tensor (equivalent of hash table)
    node_index={}
    for i,author_id in enumerate(data_df['author_id'].values):
        node_index[author_id]=i
    edge_index_tab=[]
    with open(edges_file) as fh:
        for line in fh:
            n1,n2 = line.strip().split(" ")
            try:
                i1 = node_index[int(n1)]
                i2 = node_index[int(n2)]
                edge_index_tab.append([i1,i2])
                edge_index_tab.append([i2,i1])
            except KeyError:
                continue   
    edge_index=torch.tensor(edge_index_tab,dtype=torch.long)
    # Creation of the graph and save of useful variables
    data = Data(x=x,edge_index=edge_index.t().contiguous(),y=y)
    data.train_mask = torch.tensor(train_mask,dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask,dtype=torch.bool)
    data.validation_mask = torch.tensor(validation_mask,dtype=torch.bool)
    data.hindex_validation = torch.tensor(hindex_validation,dtype=torch.float).view(-1,1)
    self.data = data
    self.testids = data_df[data_df.hindex==-1][["author_id"]]

    

# We create the graph thanks to our custom class
datagraph = AutorGraph("../../data/author-full-embeddings-"+embedding_name+".csv","../../data/coauthorship.edgelist",nb_features)
data = datagraph.data
testids = datagraph.testids

# We set the sizes of our layers
nb_hidden1 = int(sys.argv[3])
nb_hidden2 = int(sys.argv[4])
nb_hidden3 = int(sys.argv[4])
nb_classes = 1

dp = float(sys.argv[5])

if sys.argv[6]=="lstm":
  jk="lstm"
else:
  jk="cat"

# We define our GraphSAGE Network architecture
class GSAGE(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(42)
        super().__init__()
        self.conv = GraphSAGE(nb_features,nb_hidden1,2,nb_hidden2,dropout=dp,jk=jk)
        self.linear1 = nn.Linear(nb_hidden2,nb_hidden3)
        self.linear2 = nn.Linear(nb_hidden3,nb_classes)

    def forward(self, data):
      x, edge_index = data.x, data.edge_index
      # GraphSAGE network
      x = self.conv(x, edge_index)

      # Multi-layer perceptron
      x = self.linear1(x)
      x = F.relu(x)
      x = F.dropout(x, p=dp, training=self.training)
      x = self.linear2(x)
      
      return F.relu(x)

# The loss we minimize is the MSELoss
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GSAGE().to(device)


data_cuda = data.to(device)
# We use Adam optimizer with learning rate of 0.01 and weight decay of 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
torch.set_printoptions(profile="short")

# We initialize variables that will store the model with the best accuracy on the validation set
best_regression_vector = torch.tensor([],dtype=torch.float)
best_validation_MSE = 300

# Training our model over 200 epochs
model.train()
for epoch in range(200):
    print("Epoch #"+str(epoch))
    optimizer.zero_grad()
    out = model(data_cuda)
    loss = criterion(out[data_cuda.train_mask], data_cuda.y[data_cuda.train_mask])
    print("Loss on training set="+str(loss.item()))
    validation_loss = criterion(out[data_cuda.validation_mask], data_cuda.hindex_validation)
    if validation_loss.item() < best_validation_MSE:
      best_regression_vector = out[data_cuda.test_mask]
      best_validation_MSE = validation_loss.item()
    print("Loss on validation set="+str(validation_loss.item()))
    loss.backward()
    optimizer.step()

def test():
  torch.no_grad()
  regression_vector = out[data_cuda.test_mask]
  return regression_vector.detach().numpy()

# Testing the results of our model on the test set
res = test().T[0]
res_df = pd.DataFrame(res,columns=['hindex'])
output_df = pd.concat([testids.reset_index(drop=True),res_df.reset_index(drop=True)],axis=1)
output_df.to_csv("results/predictions_"+str(time.time())+".csv",index=False)