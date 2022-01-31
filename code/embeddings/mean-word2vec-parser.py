import sys
import json
import gensim.downloader as api
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

if len(sys.argv) != 3:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<model-name> <model-size>")
    exit()

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

model_name = sys.argv[1]
model_size = int(sys.argv[2])

# Load a word2vec pretrained model from gensim API
print("Loading model", model_name, "...")
wv = api.load(model_name)
print("Model was loaded.")

with open("../../data/abstracts.txt") as f:
    with open('../../data/sentence-embeddings/mean-embeddings-' + model_name +'.csv', 'w') as export_f:
        count_line = 0
        for line in f:
            # split the line after hyphens
            splited = line.split("----", 1)

            abstract_id = splited[0]
            abstract_content = json.loads(splited[1])

            length = 0
            embedding = np.zeros((model_size,))

            # for each abstract compute the weighted average of its words
            for word, occ in abstract_content["InvertedIndex"].items():
                # tokenize words once again and filter non
                for tokenized_word in [tokenized_word.lower() for tokenized_word in word_tokenize(word) if tokenized_word.isalpha()]:
                    # filter out words that are not registered in the dictionary
                    try:
                        # filter out stopwords
                        if not tokenized_word in stopwords:
                            # note that strings are lowered
                            vec_word = wv[tokenized_word]
                            length += len(occ)
                            embedding += len(occ)*vec_word
                    except KeyError:
                        continue
            embedding /= length
            # write to csv file
            export_f.write(abstract_id + "," + ','.join(['%.5f' % num for num in embedding]) + '\n')
            
            count_line+=1
            if count_line % 1000 == 0:
                print("Processed line: ", count_line)