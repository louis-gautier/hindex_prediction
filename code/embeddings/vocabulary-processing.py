import json
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

'''
This program extracts the vocabulary from the abstract dataset
---------
Tokenization is performed on each word, 
punctuation and words containing non-alphabetic characters are removed
'''

fdist = FreqDist()
with open("../../data/abstracts.txt") as f:
    count_line = 0
    for line in f:
        # split the line after hyphens
        splited = line.split("----", 1)

        abstract_id = splited[0]
        abstract_content = json.loads(splited[1])

        length = abstract_content["IndexLength"]
        
        for word, occ in abstract_content["InvertedIndex"].items():
            # tokenize 
            for tokenized_word in [tokenized_word.lower() for tokenized_word in word_tokenize(word) if tokenized_word.isalpha()]:
                fdist[tokenized_word] += len(occ)

        count_line+=1
        if count_line % 1000 == 0:
            print("Processed line: ", count_line)

    common = fdist.most_common(fdist.B())
    pd.DataFrame(common, columns=['word', 'count']).to_csv("../../data/vocabulary.csv", sep=",", index=False)

    