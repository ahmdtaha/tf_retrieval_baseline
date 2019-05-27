# Tensorflow Retrieval Baseline
This repository provides a retrieval/space embedding baseline using multiple retrieval datasets and ranking losses. 

This code is based on  [triplet-reid](https://github.com/VisualComputingInstitute/triplet-reid) repos.

Evaluation Metrics: Normalized Mutual Information (NMI), Recall@K

| Method    | Normalized | Margin | NMI   | R@1   | R@4   |
|-----------|------------|--------|-------|-------|-------|
| Semi-Hard | Yes        | 0.2    | 0.902 | 87.43 | 95.42 |