# Part of speech tagging using Viterbi algorithm
<img src="https://github.com/tonyjoo974/HMM-POS-tagging/blob/main/pos-tag.png" width="80%"></img>
<img src="https://github.com/tonyjoo974/HMM-POS-tagging/blob/main/pos-trellis.jpg" width="20%"></img>  

## Goal:
Your tagging function will be given the training data with tags and the test data without tags. Your tagger should use the training data to estimate the probabilities it requires, and then use this model to infer tags for the test input. The main mp4 function will compare these against the correct tags and report your accuracy.

## Requirements:
```
python3
numpy
```
## Running:
To run the code on the Brown corpus data you need to tell it where the data is and which algorithm to run, either baseline, viterbi_1:
```
python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm [baseline, viterbi_1]
```
