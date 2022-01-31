#!/bin/bash
models=("word2vec-google-news" "fasttext-wiki-news-subwords")
sizes=("300" "300")
pca_components=128
for i in {0..1}; do   
    python3 mean-word2vec-parser.py "${models[i]}-${sizes[i]}" "${sizes[i]}"
done
for i in {0..1}; do   
    python3 embeddings-pca-postprocessor.py "../../data/sentence-embeddings/mean-embeddings-${models[i]}-${sizes[i]}.csv" "${pca_components}" "../../data/sentence-embeddings/mean-embeddings-${models[i]}-${sizes[i]}-pca-${pca_components}.csv"
done
