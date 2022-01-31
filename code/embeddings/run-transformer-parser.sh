#!/bin/bash
models=("all-mpnet-base-v2" "allenai-specter")
pca_components=128
for i in {0..1}; do   
    python3 transformer-parser.py "${models[i]}"
done
for i in {0..1}; do   
    python3 embeddings-pca-postprocessor.py "../../data/sentence-embeddings/transformer-embeddings-${models[i]}.csv" "${pca_components}" "../../data/sentence-embeddings/transformer-embeddings-${models[i]}-pca-${pca_components}.csv"
done
