#!/bin/bash
python3 node2vec-parser.py --input ../../data/coauthorship.edgelist  --output ../../data/node-embeddings/deepwalk-128.emb
python3 node2vec-parser.py --input ../../data/coauthorship.edgelist  --output ../../data/node-embeddings/node2vec-128-p4-q1.emb --p 4 --q 1