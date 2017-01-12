# Iterative Classification Algorithm

This is a python/sklearn implementation of the Iterative Classification Algorithm from:

Qing Lu, Lise Getoor, [Link-based classification](http://linqs.cs.umd.edu/projects/projects/lbc/) (ICML 2003)

which served as a semi-supervised classification baseline in our recent paper:

Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (2016)

This implementation is largely based on and adapted from: [https://github.com/sskhandle/Iterative-Classification](https://github.com/sskhandle/Iterative-Classification)

## Installation

```bash
python setup.py install
```

## Requirements
* sklearn
* networkx

## Run the demo

```bash
python train.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* a N by E binary label matrix (E is the number of classes).

Have a look at the `load_data()` function in `utils.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/. In our version (see `data` folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016). 

You can specify a dataset as follows:

```bash
python train.py -dataset citeseer
```

(or by editing `train.py`)
