import numpy as np
import pandas as pd
import time
import sys

if len(sys.argv) < 2:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<timestamp>")
    exit()

results_file = "../../data/predictions"+str(sys.argv[1])+".csv"

# Getting the predictions and the other files needed
result_df = pd.read_csv(results_file)
excluded_df = pd.read_csv("../../data/easyprediction.csv")
excluded = excluded_df[["author"]].values.T[0]
attention_df = pd.read_csv("../../data/attentionprediction.csv")
attention = attention_df[["author"]].values.T[0]
test_df = pd.read_csv("../../data/test.csv")


for i in range(len(test_df.index)):
    author_id = test_df.loc[i,"author"]
    # If the considered author is part of excluded, he wrote only one paper and we assign it a hindex of 1
    if(author_id in excluded):
        hindex = 1
    else:
        hindex = result_df[result_df.author_id==author_id]["hindex"].values[0]
        # If the considered author is part of attention, he wrote 2,3 or 4 paper
        if(author_id in attention):
            max_hindex = attention_df[attention_df.author==author_id]["nbpapers"].values[0]
            # If the model assigned it a h-index of more than the maximum h-index for this author, we bring this h-index back to the maximum possible value
            if(hindex>max_hindex):
                hindex=max_hindex
    test_df.at[i,"hindex"] = hindex
test_df=test_df[["author","hindex"]]

# We save the results in a csv file
test_df.to_csv("results/test"+str(time.time())+".csv",index=True)