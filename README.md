# inf554_challenge

In order to run all the pre-processing algorithms and the models, please download `abstracts.txt`, `author_papers.txt`, `train.csv`, `test.csv` and `coauthorship.edgelist`  in the `data` folder. Then, create a virtual environment, activate it and install the required libraries with
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Once it is done, you can start pre-processing the data to create the necessary embeddings.

## Pre-processing
The first step of pre-processing is to extract the filtered vocabulary from the `abstracts.txt` file. This can be done by running
```bash
cd code/embeddings
python3 vocabulary-processing.py
```

Then, you can extract full sentences (for supervised embeddings including *SBERT* and *InferSent*) from the `abstract.txt` file with 
```bash
cd code/embeddings
python3 sentence-builder.py
```

To create mean *word2vec*-like embeddings (*word2vec*, *GloVe*, *fastText*), you can run the provided `.sh` script.
```bash
cd code/embeddings
./run-meanvec-parser.sh
```

To create *InferSent* embeddings (DEPRECATED)
```bash
cd code/embeddings
./prepare-infersent-parser.sh
./run-infersent-parser.sh
```

To create *SBERT* embeddings
```bash
cd code/embeddings
./run-transformer-parser.sh
```

Finally, to create *node2vec* embeddings
```bash
cd code/embeddings
./create-node2vec.sh
```

Finally, to merge embeddings and create the full features that will be stored in `data/author-embeddings`
```bash
cd code/embeddings
./merge-embeddings.sh
```

## Models

### Multilayer perceptron (MLP)
To perform **k-fold** cross validation with different sets of hyperparameters on a given *author-embeddings*, do
```bash
cd code/MLP
python3 MLP-k-fold.py <author-embeddings>
```
The logs obtained with SPECTER transformer-based embeddings reduced to 128 dimensions and concatenated with deepwalk embeddings of size 128 are provided in `data/logs/allenai-specter-pca-128-deepwalk-128-logs.txt`.

To perform training with a multilayer perceptron with chosen parameters
```bash
cd code/MLP
python3 MLP-pipeline.py <author-embeddings> <n_hidden1> <n_hidden2> <n_epochs> <dropout_p> <learning-rate> <test-output> [<plot-data>]
```

If you exported plot data, you can display it thanks to `MLP-figure.py`
```bash
cd code/MLP
python3 MLP-figure.py <output-figure> <input-losses> [<other-input-losses> ...]
```

### XG-Boost

To perform **k-fold** cross validation with different sets of hyperparameters on a given *author-embeddings*, do
```bash
cd code/xgboost
python3 xgboost-cross_validation.py <author-embeddings> <n_estimators> <max_depth> <learning-rate> <subsample> <k-fold>
```

To perform training with chosen parameters
```bash
cd code/xgboost
python3 xgboost-submission.py <author-embeddings> <n_estimators> <max_depth> <learning-rate> <subsample> <k-fold> <submission_csv>
```

### Graph Neural Networks

Once you have an abstract embeddings file ready, execute the following instructions to compute node embeddings (not taking graph node metrics into consideration yet):
```bash
cd code/GNN
python3 abstracts_to_author.py <abstract-embeddings-file> <abstract-embeddings-size>
```

To add node metrics to the author embeddings file
```bash
cd code/GNN
python3 graph_metrics_features.py <author-embeddings-file> <abstract-embeddings-size>
```

Choose your model among GCN, GAT and GraphSAGE and execute one of those three lines:
```bash
cd code/GNN
python3 pipeline_GCN.py <author-full-embeddings-file> <author-full-embeddings-size> <size-layer1> <size-layers2-3> <dropout-rate>
python3 pipeline_GAT.py <author-full-embeddings-file> <author-full-embeddings-size> <size-layer1> <size-layers2-3> <dropout-rate>
python3 pipeline_GraphSAGE.py <author-full-embeddings-file> <author-full-embeddings-size> <size-layer1> <size-layers2-3> <dropout-rate> <aggregation-function>
```

### Post-processing

Execute the following instruction for post-processing
```bash
cd code/GNN
python3 postprocessing.py <predictions-timestamp>
```