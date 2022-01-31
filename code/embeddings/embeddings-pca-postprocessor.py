import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle as pk
'''
This program loads sentence embeddings and performs pca with scaling
---------
SentenceTransformer embeddings are computed thanks to pre-trained models
'''

if len(sys.argv) < 4:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<sentence-embeddings> <pca-dimension> <output-file>")
    exit()

print("Loading sentence embeddings...")
embeddings_df = pd.read_csv(sys.argv[1], header=None, delimiter=',')
embeddings_df.rename(columns={0 :'abstract_id'}, inplace=True)
embeddings_df.set_index('abstract_id', inplace=True)
embeddings_df.dropna(inplace=True)
print(embeddings_df.head())

pca_dim = int(sys.argv[2])

output_file = sys.argv[3]

print("Scaling data...")
scaler = StandardScaler()
embeddings_df = pd.DataFrame(scaler.fit_transform(embeddings_df.values), columns=embeddings_df.columns, index=embeddings_df.index)
print(embeddings_df.head())

print("Performing PCA...")
pca = PCA(n_components=pca_dim)
pca.fit(embeddings_df)

embeddings_df_pca = pd.DataFrame(pca.transform(embeddings_df.values), index=embeddings_df.index)
embeddings_df_pca.to_csv(output_file, sep=',', float_format='%.5f', header=False)
print(embeddings_df_pca.head())