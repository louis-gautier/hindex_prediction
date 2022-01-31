import math
import sys
import numpy as np
import pandas as pd

# Set the parameters of the model here
if len(sys.argv) < 4:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<abstract-embeddings> <node-embeddings> <output-path>")
    exit()

# 1 - calculer moyenne / écart type des embeddings non nan
# 2 - remplacer les papiers manquants
abstract_embeddings = sys.argv[1]
node_embeddings = sys.argv[2]
output_path = sys.argv[3]

abstracts_df=pd.read_csv(abstract_embeddings,header=None)
abstracts_df.rename(columns={0 :'abstract_id'}, inplace=True)
means = abstracts_df.loc[:,abstracts_df.columns!="abstract_id"].dropna().mean().values
stds = abstracts_df.loc[:,abstracts_df.columns!="abstract_id"].dropna().std().values
# we set indexes on the abstract id to speed up requests later on
abstracts_df.set_index('abstract_id', inplace=True)
abstracts_df.sort_index(inplace=True)
abstract_embeddings_size = abstracts_df.shape[1]

nodes_df=pd.read_csv(node_embeddings, delimiter=' ', skiprows=1, header=None)
nodes_df.rename(columns={nodes_df.columns[0]:'author'}, inplace=True)
nodes_df.set_index('author', inplace=True)
nodes_df.sort_index(inplace=True)
node_embeddings_size = nodes_df.shape[1]
no_nodes = 0

with open("../../data/author_papers.txt") as authors:
    with open(output_path, 'w') as export_f:
            
        count_line = 0
        notvalid_authors = 0
        for line in authors:
            # split the line after ":"
            splited = line.split(":", 1)

            author_id = splited[0]
            author_papers = splited[1].strip().split("-")
            
            nb_papers = len(author_papers)
            author_embedding = np.zeros((abstract_embeddings_size,))
            # for each author compute the average of its abstracts
            for paper in author_papers:
                try:
                    paper_embedding = abstracts_df.loc[int(paper),:]
                    if(math.isnan(paper_embedding.values[0])):
                        paper_embedding = np.zeros((abstract_embeddings_size,))
                        nb_papers-=1
                except KeyError:
                    paper_embedding = np.zeros((abstract_embeddings_size))
                    nb_papers-=1
                author_embedding += paper_embedding
            if(nb_papers>0):    
                author_embedding /= nb_papers
            else:
                # prendre toutes les coordonnées au hasard (en suivant la même distribution que les autres?)
                author_embedding = means + stds*np.random.randn(abstract_embeddings_size)
                notvalid_authors += 1
            try:
                feature_embedding = nodes_df.loc[int(author_id), :]
            except KeyError:
                feature_embedding = np.zeros((node_embeddings_size))
                no_nodes += 1
            #print(feature_embedding.shape)
            # write to csv file
            export_f.write(author_id + "," + ','.join(['%.5f' % num for num in author_embedding]) + ',' + ','.join(['%.5f' % num for num in feature_embedding])  +'\n')
            
            count_line+=1
            if(count_line%10000==0):
                print("Processed lines: ", count_line)

print(notvalid_authors, "are non valid authors")
# 1884 non valid authors
print(no_nodes, "have no node embeddings")
