# Watch Your Back: Identifying Cybercrime Financial Relationships in Bitcoin through Back-and-Forth Exploration

This is the repository for *Kusarikoma*, a software tool that implements the methodology described in our [CCS'22 paper](https://doi.org/10.1145/3548606.3560587). It automatically explores addresses of the Bitcoin blockchain (the seeds), produces an address-transaction directed graph, and prevents explosion within the resulting graph. Finally, it is able to find relationships between targeted addresses. It receives as input:

- Blockchain data parsed using [BlockSci](https://github.com/citp/BlockSci)
- A set of tags for Bitcoin addresses and clusters (located in data/tagging/)
- A set of Bitcoin addresses to explore (the seeds)

Kusarikoma iterates through all deposit and withdrawal transactions of the seeds, looking for all input and output addresses in each transaction. Transactions that could introduce false relations like CoinJoin or Force-address-reuse are filtered out. For each address found, Kusarikoma tries to determine if it is an address owned by a service (service-address) or not. Service-addresses are not explored further. Non-service-addresses are added to a worklist to be explored in the next iteration. The exploration finishes when there are no more addresses to explore in the worklist, or when an optional limit is reached (a given number of steps, maximum number of nodes, or maximum explosion rank).

In order to classify an address, Kusarikoma uses a set of tags which specify the owner (i.e. who owns the cluster) and the user (the beneficiary) of the address. For addresses without tags, Kusarikoma makes use of its AI (an exchange-classifier) to determine if the address is possibly owned by an exchange. If so, the address is considered a service-address.

Kusarikoma also makes use of three additional classifiers (oracles) to determine if an address found within the exploration can be linked to four different malicious operations: cerber, pony, skidmap, or glupteba.

More details can be found in our [CCS'22 paper](https://doi.org/10.1145/3548606.3560587).


## Quick start

1) Install python dependencies, e.g. using a conda environment:

```
conda create --name Kusarikoma -c conda-forge --file requirements.txt
```

2) Install [BlockSci](https://github.com/citp/BlockSci) version v0.7.0, and parse the full (or up to a partial height) Bitcoin blockchain, as specified [here](https://citp.github.io/BlockSci/setup.html)

3) Build the tag database:

```
cd code/
python lib/tags.py --build --blocksci path/to/blocksci.cfg --height 716600
```

4) Explore the seeds to build a graph:

```
python explore.py --blocksci path/to/blocksci.cfg --seed 1BkeGqpo8M5KNVYXW3obmQt1R58zXAqLBQ --seed 1CeLgFDu917tgtunhJZ6BA2YdR559Boy9Y --seed 19hi8BJ7HxKK45aLVdMbzE6oTSW5mGYC82 --seed 1N9ALZUgqYzFQGDXvMY5j1c7PGMMGYqUde --oracle pony --steps 3 --estimator exchange-classifier.joblib --height 716600 --maxexprank 2000 --output pony_exploration
```

5) Search for relations within the graph:

```
python lib/graph.py --graph pony_exploration.gml --directed --decorate path/to/blocksci.cfg
```


## Exchange-classifier

To produce an exchange-classifier, create a set of positive and negative Bitcoin addresses, extract features from them, and train/test a model.

1) Each set must be a tab-separated-values (tsv) file with a header row. It must contain at least an "address" field, e.g.:

```
address	name
1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s	binance
```

2) Extract features from each set of addresses. Then merge all feature vectors into data/datasets/dataset.tsv:

```
cd code/
python lib/blockchain_feature_extraction.py --inputfile positive_addresses.tsv --output positive_addresses_features.tsv --height 716600 --blocksci path/to/blocksci.cfg --label exchange
python lib/blockchain_feature_extraction.py --inputfile negative_addresses.tsv --output negative_addresses_features.tsv --height 716600 --blocksci path/to/blocksci.cfg --label non-exchange
cp positive_addresses_features.tsv ../data/datasets/dataset.tsv
tail -n+2 negative_addresses_features.tsv >> ../data/datasets/dataset.tsv
```

3) Build the model using:

```
python binary_classifiers.py
```


## Clustering

BlockSci multi-input clusters are used during the tag database creation, and during the exploration. The clusters will be created or loaded from a predefined path, built into the same folder than the BlockSci config file specified, as follows:

```
{os.path.dirname(config_file)}/clusters/multi-input{('_' + str(height+1)) if height else ''}
```

So, for height=716600, the clusters path will be created in:

```
{os.path.dirname(config_file)}/clusters/multi-input_716601
```

While for height=0, the resulting path will be:

```
{os.path.dirname(config_file)}/clusters/multi-input
```

Kusarikoma will automatically create and load the clusters from such paths, transparent to the user. Be careful when using too many different heights, due to the disk space usage.


## Tags

All tag files should be located in data/tagging/ folders, and must follow the next format:

- CSV file, no headers, no index, with six columns: address, ticker, category, tag, subtype, and url
- address, ticker, category and url are mandatory; tag and subtype can be empty


## Paths

The path search algorithm will look for either directed (--directed) or undirected (--undirected) paths between the entities in the graph. However, as stated in the [paper](https://doi.org/10.1145/3548606.3560587), meaningful relations are those that describe money flow, represented by directed paths. The argument --decorate can be used to print blockchain properties of addresses and transactions.


## Oracles

Oracles can be invoked using --oracle [cerber|pony|glupteba]; Addresses detected by the oracle will be purple-colored in the resulting graph.


