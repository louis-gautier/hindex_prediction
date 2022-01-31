import sys
import json
import numpy as np
import torch
from nltk.tokenize import word_tokenize

'''
This program makes sentences from the abstract dataset
---------
Tokenization is performed on each word, 
punctuation and words containing non-alphabetic characters are removed
'''

with open("../../data/abstracts.txt") as f:
    with open('../../data/abstracts-sentences.csv', 'w') as export_f:
        count_line = 0
        for line in f:
            # split the line after hyphens
            splited = line.split("----", 1)

            abstract_id = splited[0]
            abstract_content = json.loads(splited[1])

            length = abstract_content["IndexLength"]

            sentence_list = [[] for _ in range(length)]
            for word, occ in abstract_content["InvertedIndex"].items():
                for i in occ:
                    for tokenized_word in [tokenized_word.lower() for tokenized_word in word_tokenize(word) if tokenized_word.isalpha()]:
                        sentence_list[i].append(tokenized_word)
            sentence = ""
            for group_words in sentence_list:
                for word in group_words:
                    sentence += word + " "

            # write to csv file
            export_f.write(abstract_id + "," + sentence + '\n')
            
            count_line+=1
            if count_line % 1000 == 0:
                print("Processed line: ", count_line)
