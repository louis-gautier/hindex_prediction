import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle as pk
from sentence_transformers import SentenceTransformer, util
'''
This program loads pre-filtered sentences and turns them into SentenceTransformer embeddings
---------
SentenceTransformer embeddings are computed thanks to pre-trained models
'''

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = "all-mpnet-base-v2"

print("Loading sentences...")
abstract_df = pd.read_csv("../../data/abstracts-sentences.csv", header=None, delimiter=',')

print("Loading model", model_name, "...")
model = SentenceTransformer(model_name)
print("SentenceTransformer model was loaded.")

export_name = '../../data/sentence-embeddings/transformer-embeddings-' + model_name +'.csv'

with open(export_name, 'w') as export_f:
    count_line = 0
    for index, row in abstract_df.iterrows():

        abstract_id = int(row[0])
        
        sentence = str(row[1])

        embedding = model.encode(sentence)

        # write to csv file
        export_f.write(str(abstract_id) + ',' + ','.join(['%.5f' % num for num in embedding]) + '\n')
        count_line += 1
        if count_line % 1000 == 0:
            print("Processed line: ", count_line)