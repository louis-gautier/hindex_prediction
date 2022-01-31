#!/bin/bash
mkdir ../../data/GloVe
curl -Lo ../../data/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip ../../data/GloVe/glove.840B.300d.zip -d ../../data/GloVe/
mkdir ../../data/fastText
curl -Lo ../../data/fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip ../../data/fastText/crawl-300d-2M.vec.zip -d ../../data/fastText/

mkdir ../../data/encoder
curl -Lo ../../data/encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
curl -Lo ../../data/encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl