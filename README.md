# Tensorflow Retrieval Baseline
This repository provides a retrieval/space embedding baseline using multiple retrieval datasets and ranking losses. This code is based on  [triplet-reid](https://github.com/VisualComputingInstitute/triplet-reid) repos.

### Evaluation Metrics
1. Normalized Mutual Information (NMI)
2. Recall@K

### Deep Fashion In-shop Retrieval Evaluation
All the following experiments assume a training mini-batch of size 60. The architecture employed is the one used in [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737) but ResNet is replaced by a DenseNet169.
Optimizer: Adam, Number of iterations = 25K

| Method    | Normalized | Margin | NMI   | R@1   | R@4   | # of classes | #samples per class |
|-----------|------------|--------|-------|-------|-------|--------------|--------------------|
| [Semi-Hard](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss) | Yes | 0.2    | 0.902 | 87.43 | 95.42 | 10| 6|
| Hard-Negative | No | 1.0    | 0.904 | 88.38 | 95.74 | 10| 6|
| [Lifted Structured](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/lifted_struct_loss) | No | 1.0    | 0.903 | 87.32 | 95.59 | 10| 6|
| [N-Pair Loss](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/npairs_loss) | No | N/A    | 0.903 | 89.12 | 96.13 | 2| 30|
| [Angular Loss](https://github.com/geonm/tf_angular_loss) | Yes | N/A  | 0.8931 |  84.70 | 92.32 | 2| 30|
| Custom [Contrastive Loss](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/contrastive_loss) | Yes | 1.0  | 0.826 |  44.09 | 67.17 | 4| 15|

### CUB200-2011 Retrieval Evaluation
Mini-batch size=120. Architecture: Inception_Net V1.
Optimizer: Momentum. Number of iterations = 10K

| Method    | Normalized | Margin | NMI   | R@1   | R@4   | # of classes | #samples per class |
|-----------|------------|--------|-------|-------|-------|--------------|--------------------|
| [Semi-Hard](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss) | Yes | 0.2    | 0.587 | 49.03 | 73.43 | 20| 6|
| Hard Negatives | No | 1.0    | 0.561 | 46.55 | 71.03 | 20| 6|
| [Lifted Structured](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/lifted_struct_loss) | No | 1.0    | 0.502 | 35.26 | 59.82 | 20| 6|
| [N-Pair Loss](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/npairs_loss) | No | N/A    | 0.573 | 46.52 | 59.26 | 60| 2|
| [Angular Loss](https://github.com/geonm/tf_angular_loss) | Yes | N/A    | 0.546 | 45.50 | 68.43 | 60 | 2|
| Custom [Contrastive Loss](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/contrastive_loss) | Yes | 1.0    | 0.476 | 37.27 | 62.39 | 30| 4|

### Stanford Online Products Retrieval Evaluation
Mini-batch size=120. Architecture: Inception_Net V1.
Optimizer: Adam. Number of iterations = 30K

| Method    | Normalized | Margin | NMI   | R@1   | R@4   | # of classes | #samples per class |
|-----------|------------|--------|-------|-------|-------|--------------|--------------------|
| Hard Negatives | No | 1.0    | 0.895 | 72.03 | 82.55 | 20| 6|

### Requirements
* Python 3+ [Tested on 3.4.7]
* Tensorflow 1+ [Tested on 1.8]

### Code Setup
1. Update the directories' paths in constants.py 
2. train.py
3. embed.py
4. eval.py

### TODO
* [TODO] bash script for train, embed and then eval
* [TODO] Explain the fast contrastive loss sampling procedure
* [TODO] Explain why contrastive loss is different from [tf.contrib.losses.metric_learning.contrastive\_loss](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/contrastive_loss)
* [TODO] Evaluate on CUB (a small dataset) with a small architecture like ResNet
* [TODO] Evaluate space embedding during training.
