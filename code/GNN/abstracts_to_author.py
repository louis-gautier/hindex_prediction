# This file generates author embeddings only taking the content of their top-cited abstracts in consideration

import math
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 3:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<abstract-embeddings> <model-size>")
    exit()

# Get the parameters of the abstract embeddings model
model_name = str(sys.argv[1])
model_size = int(sys.argv[2])
abstract_embeddings = "../../data/"+model_name+".csv"

# Get the abstract embeddings and compute their mean and standard deviation for each column
abstracts_df = pd.read_csv(abstract_embeddings,header=None)
abstracts_df.rename(columns = {0 : 'abstract_id'}, inplace=True)
means = abstracts_df.loc[:,abstracts_df.columns != "abstract_id"].dropna().mean().values
stds = abstracts_df.loc[:,abstracts_df.columns != "abstract_id"].dropna().std().values

# We set an index on the abstract id to speed up requests later on
abstracts_df.set_index('abstract_id', inplace=True)
abstracts_df.sort_index(inplace=True)

# We get train data for node labels
labels_df = pd.read_csv('../../data/train.csv')
# We get test data, and w assign them an h-index of -1
test_df = pd.read_csv('../../data/test.csv')[["author","hindex"]]
test_df = test_df.replace([np.nan], [-1])

# Dataframe combining train and test data
authors_df = pd.concat([labels_df,test_df],ignore_index=True)
authors_df.set_index('author', inplace=True)
authors_df.sort_index(inplace=True)

# Dataframe containing all authors from the test set with only one paper
excluded_df = pd.DataFrame(columns=["author"])
# Dataframe containing all authors from the test set with 2,3 or 4 papers
attention_df = pd.DataFrame(columns=["author","nbpapers"])

with open("../../data/author_papers.txt") as authors:
    with open('../../data/author-embeddings-' + model_name +'.csv', 'w') as export_f:    
        count_line = 0
        # Authors considered as not valid are those for which we don't have any paper embedding available
        notvalid_authors = 0
        for line in authors:
            # Split the line after ":"
            splited = line.split(":", 1)

            # Get the author_id and the author's papers
            author_id = splited[0]
            author_papers = splited[1].strip().split("-")
            nb_papers = len(author_papers)

            # Get the h-index of the author (-1 if the author is in the test set)
            try:
                hindex = authors_df.loc[int(author_id),"hindex"]
            except KeyError:
                continue

            # If the author has only one paper, we can ignore it and save it for later if it is part of the test set
            if(nb_papers==1):
              if(hindex==-1):
                  excluded_df = excluded_df.append({"author": author_id},ignore_index=True)
              continue
            else:
              # If the author has 2,3 or 4 papers and is part of the test set, we save this information for later
              if(nb_papers<5 and hindex==-1):
                attention_df = attention_df.append({"author": author_id,"nbpapers":nb_papers},ignore_index=True)
            
            # Initialization of the author embedding
            author_embedding = np.zeros((model_size,))
            
            # We compute the mean of the embeddings of available abstracts
            for paper in author_papers:
                try:
                    paper_embedding = abstracts_df.loc[int(paper),:]
                    if(math.isnan(paper_embedding.values[0])):
                        paper_embedding = np.zeros((model_size,))
                        nb_papers-=1
                except KeyError:
                    # If we have no stored embedding for this paper, we ignore it
                    paper_embedding = np.zeros((model_size))
                    nb_papers-=1
                author_embedding += paper_embedding
            if(nb_papers>0):   
                author_embedding /= nb_papers
            else:
                # In this case, we don't have any paper for the author, we thus generate one similar to the one of all papers (following a normal distribution with the same characteristics than the distribution of paper embeddings)
                author_embedding = means + stds*np.random.randn(model_size)
                notvalid_authors += 1
            
            # Write to csv output file
            export_f.write(author_id + "," + ','.join(['%.5f' % num for num in author_embedding]) + ',' + str(hindex) +'\n')
            
            count_line+=1
            if(count_line%10000==0):
                print("Processed lines: ", count_line)

# Number of authors for which we don't have any article
print(notvalid_authors)

# We save our 2 supplementary dataframes to csv
excluded_df.to_csv("../../data/easyprediction.csv")
attention_df.to_csv("../../data/attentionprediction.csv")