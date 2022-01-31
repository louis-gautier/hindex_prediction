#!/bin/bash
models=("1" "2")
w2v=("../../data/GloVe/glove.840B.300d.txt" "../../data/fastText/crawl-300d-2M.vec")
for i in {0..1}; do   
    python3 infersent-parser.py "${models[i]}" "${w2v[i]}"
done