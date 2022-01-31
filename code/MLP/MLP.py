
import torch
from torch import nn
import torch.nn.functional as F

class TunableParameters:
    def __init__(self, n_hidden1, n_hidden2, use_dropout, dropout_p, lr):
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.lr = lr
    
    def print_parameters(self):
        print("Tunable parameters:")
        print("########################################")
        print("Size of hidden layer 1:", self.n_hidden1)
        print("Size of hidden layer 2:", self.n_hidden2)
        print("Using dropout:", self.use_dropout)
        print("Dropout p:", self.dropout_p)
        print("Learning rate:", self.lr)

    def get_param_string(self):
        param_string = 'h1-' + str(self.n_hidden1) + '-h2-' + str(self.n_hidden2)
        if self.use_dropout:
            param_string += f'-d-{self.dropout_p}'
        param_string += f'-lr-{self.lr}'
        return param_string
 
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, use_dropout=True, dropout_p=0.2):
        super(MLP, self).__init__()
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.fc1 = torch.nn.Linear(n_input, n_hidden1)
        self.fc2 = torch.nn.Linear(n_hidden1, n_hidden2)
        if use_dropout:
            self.dropout1 = torch.nn.Dropout(p=self.dropout_p)
            self.dropout2 = torch.nn.Dropout(p=self.dropout_p)
        self.output = torch.nn.Linear(n_hidden2, n_output)  
        
    def forward(self, x, verbose=False):
        x = self.fc1(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = F.relu(x)
        x = self.output(x)
        return x