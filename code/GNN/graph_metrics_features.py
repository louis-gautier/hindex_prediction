import pandas as pd
import sys
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain

if len(sys.argv) < 3:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<author-embeddings> <model-size>")
    exit()

# Get the author embeddings (based on abstracts only)
embedding_name=str(sys.argv[1])
nb_features = int(sys.argv[2])
data_df = pd.read_csv("../../data/author-embeddings-"+embedding_name+".csv",header=None)
data_df.rename(columns={0 :'author_id', (nb_features+1):'hindex'}, inplace=True)

# Get the graph structure
edges_file="../../data/coauthorship.edgelist"

# We build a hash table (dictionary) giving the index in data_df associated to every author_id
node_index={}
for i,author_id in enumerate(data_df['author_id'].values):
    node_index[author_id]=i

# We will build the graph based on its edges. We only add edges for which both ends are in the train or in the test set
edge_index_tab=[]
with open(edges_file) as fh:
    for line in fh:
        n1,n2 = line.strip().split(" ")
        try:
            i1 = node_index[int(n1)]
            i2 = node_index[int(n2)]
            edge_index_tab.append((int(n1),int(n2)))
        except KeyError:
            continue

# Creation of the graph with Networkx
G = nx.Graph()
G.add_nodes_from(data_df['author_id'].values)
G.add_edges_from(edge_index_tab)
print(nx.info(G))

# Definition of the node metrics except those linked to partitions
degree = [val for (node,val) in nx.degree(G)]
neigh_average_degree = list(nx.average_neighbor_degree(G).values())
pgrank = list(nx.pagerank(G).values())
cre_number = list(nx.core_number(G).values())

# Computation of the partition
partition = list(community_louvain.best_partition(G).values())
# Creating one-hot encoding of communities for each author
communities_df = pd.get_dummies(pd.Series(partition))
communities_sizes = communities_df.sum(0).values

# Initializing the metrics linked to communities

# Number of distinct communities directly linked to a node
nb_linked_communities = np.zeros(G.number_of_nodes())
# Cardinality of the distinct communities directly linked to a node
card_linked_communities = np.zeros(G.number_of_nodes())
# Cardinality of the communities directly linked to a node added as many times as their multiplicity
multi_card_linked_communities = np.zeros(G.number_of_nodes())
# Ratio between the number of neighbors in a node's commnunity and its degree
ratio_linked_communities = np.zeros(G.number_of_nodes())

# We iteratively compute these 4 metrics
for i,node in enumerate(list(G.nodes())):
  node_com = communities_df.loc[i,:].dot(communities_df.columns)
  neighbors = list(G.neighbors(node))
  linked_communities=set()
  nb_neighbors_same_com=0
  card_linked_coms=0
  for neighbor in neighbors:
    neighbor_idx = node_index[neighbor]
    neighbor_com = communities_df.loc[neighbor_idx,:].dot(communities_df.columns)
    linked_communities.add(neighbor_com)
    if(neighbor_com==node_com):
      nb_neighbors_same_com+=1
    card_linked_coms += communities_sizes[neighbor_com]
  nb_linked_communities[i]=len(linked_communities)
  multi_card_linked_communities[i]=card_linked_coms
  card_linked_communities[i]=np.sum([communities_sizes[com] for com in linked_communities])
  if len(list(neighbors))>0:
    ratio_linked_communities[i]=nb_neighbors_same_com/len(list(neighbors))
  else:
    ratio_linked_communities[i]=1

# We concatenate these 4 metrics in a single dataframe
community_metrics_df = pd.concat([pd.Series(nb_linked_communities),pd.Series(multi_card_linked_communities),pd.Series(card_linked_communities),pd.Series(ratio_linked_communities)],axis=1)

# We get the h-index of each author (-1 if it is in the test set)
labels_df = data_df[["hindex"]]

# We output all these metrics and the hindex in a single csv files
data_df = data_df.loc[:,data_df.columns!="hindex"]
to_write_df = pd.concat([data_df.reset_index(drop=True),community_metrics_df,pd.Series(degree),pd.Series(neigh_average_degree),pd.Series(pgrank),pd.Series(cre_number),labels_df.reset_index(drop=True)], axis=1)
to_write_df.to_csv("../../data/author-full-embeddings-"+embedding_name+".csv",index=False,header=None,float_format='%.6f')