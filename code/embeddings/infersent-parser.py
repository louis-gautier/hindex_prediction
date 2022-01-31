import sys
import numpy as np
import torch
import pandas as pd
import nltk
nltk.download('punkt')

'''
This program loads pre-filtered sentences and turns them into Infersent embeddings
---------
Infersent embeddings are computed thanks to pre-trained models and the pre-computed vocabulary
'''

print("Loading sentences...")
abstract_df = pd.read_csv("../../data/abstracts-sentences.csv", header=None, delimiter=',')

print("Loading vocabulary...")
vocab_data = pd.read_csv("../../data/abstracts-vocabulary.csv", header=None, delimiter=',')
vocab_df = pd.DataFrame(vocab_data, columns=['word', 'count'])
vocab_list = vocab_df['word'].tolist()

print("Loading models...")
from infersent import InferSent
if len(sys.argv) < 3:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<model-version> <w2v-path>")
    exit()
else:
    V = int(sys.argv[1])
    W2V_PATH = sys.argv[2]

MODEL_PATH = '../../data/encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
infersent.set_w2v_path(W2V_PATH)
# infersent.build_vocab_k_words(K=100000)
infersent.build_vocab_from_words(vocab_list)
print("Infersent model was loaded.")

with open('../../data/sentence-embeddings/infersent-embeddings-' + str(V) +'.csv', 'w') as export_f:
    count_line = 0
    for index, row in abstract_df.iterrows():

        abstract_id = row[0]
        
        sentence = str(row[1])

        embeddings = infersent.encode([sentence], tokenize=True, is_list=False)

        # write to csv file
        export_f.write(str(abstract_id) + ',' + ','.join(['%.5f' % num for num in embeddings[0]]) + '\n')
        count_line += 1
        if count_line % 1000 == 0:
            print("Processed line: ", count_line)