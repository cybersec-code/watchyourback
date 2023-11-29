# Watch Your Back: A Bitcoin Transaction Tracing and Revenue Estimation Platform

This is the repository for *WatchYourBack*, a platform that given a set of seed Bitcoin addresses allows to (1) trace the transactions from the seeds until they reach known addresses (e.g., exchanges) and (2) estimate the revenue received by the seed owners. 

The exploration tool implements the approach in our [ACM CCS'22 paper](https://doi.org/10.1145/3548606.3560587).

The estimation tool implements the approach in our [ACM CCS'23 paper](https://doi.org/10.1145/3576915.3623094).

You can find references to both papers at the end of this README.


## Setup

The following steps need to be performed before running the exploration or the estimation tools.

1) Install python dependencies, e.g. using a conda environment:

```
conda create --name WatchYourBack -c conda-forge --file requirements.txt
```

2) Install [BlockSci](https://github.com/citp/BlockSci) version v0.7.0, and parse the full (or up to a partial height) Bitcoin blockchain, as specified [here](https://citp.github.io/BlockSci/setup.html)

3) Build the multi-input clustering and tag database (up to the highest block parsed by BlockSci in the previous step):

```
cd code/
python lib/tags.py --build --blocksci path/to/blocksci.cfg
```

If you want to build the multi-input clustering and tags at a height **lower** than the highest block parsed from the blockchain, you can specify the --height option:

```
python lib/tags.py --build --blocksci path/to/blocksci.cfg --height 716600
```


## Estimation

At the high level, given a set of seed addresses, the estimation tool calculates the amount of BTC and USD received by the seeds using different methodologies.

At a lower level, the estimation gets all the direct deposits (DD) to the set of seeds, and according to the selected methodology, it can expand the seeds using multi-input clustering (and optionally using the change-address heuristic), and a double-counting (DC) filtering.
 
The estimation takes the following inputs:

- The blockchain data parsed using [BlockSci](https://github.com/citp/BlockSci) using the --blocksci option
- A set of Bitcoin addresses to explore (the seeds), specified using the --seeds-file option
- The methodology, with the format DD[[-OW]+[MI|MICA]][-DC], e.g., DD-DC or DD-OW+MI-DC
- An optional block height on which the exploration runs, specified using the --height option. If this option is not provided, the highest parsed height is used. 
- A set of tags for Bitcoin addresses and clusters (located in data/tagging/), already computed in step 2 of the Setup section (no need to specify in command line)

Here is a sample command line to estimate a set of seeds specified in a seeds.csv file, using the DD-OW+MI-DC methodology:

```
python estimate.py --blocksci /data/BlockSci/btc.cnf --seeds-file seeds.csv --estimation DD-OW+MI-DC -O output/
```
The above command will generate an output file with the results of the estimation in both BTC and USD. The conversion rate used is on the day of the deposits.


## Exploration

At the high level, given a set of seed addresses, the exploration produces an address-transaction directed graph, where the flow of BTCs to external entities such as exchanges can be identified. 

At a lower level, the exploration iterates through all deposit and withdrawal transactions of the seeds, looking for all input and output addresses in each transaction.
Transactions that could introduce false relations like CoinJoin or Force-address-reuse are filtered out.
For each address found, WatchYourBack tries to determine if it is an address owned by a service or not. Service-addresses are not explored further.
Non-service-addresses are added to a worklist to be explored in the next iteration. 
The exploration finishes when there are no more addresses to explore in the worklist, or when an optional limit is reached (a given number of steps, maximum number of nodes, or maximum explosion rank).
More details can be found in our [CCS'22 paper](https://doi.org/10.1145/3548606.3560587).

The exploration takes the following inputs:

- The blockchain data parsed using [BlockSci](https://github.com/citp/BlockSci) using the --blocksci option
- A set of Bitcoin addresses to explore (the seeds), specified using the --seed option that can be provided multiple times
- A set of tags for Bitcoin addresses and clusters (located in data/tagging/), already computed in step 2 of the Setup section (no need to specify in command line)
- A maximum number of exploration steps, specified using the --steps option
- An optional block height on which the exploration runs, specified using the --height option. If this option is not provided, the highest parsed height is used. 
- An optional output prefix used to name the different output files

Here is a sample command line to explore 4 seeds belonging to the Pony malware family:

```
python explore.py --blocksci path/to/blocksci.cfg --seed 1BkeGqpo8M5KNVYXW3obmQt1R58zXAqLBQ --seed 1CeLgFDu917tgtunhJZ6BA2YdR559Boy9Y --seed 19hi8BJ7HxKK45aLVdMbzE6oTSW5mGYC82 --seed 1N9ALZUgqYzFQGDXvMY5j1c7PGMMGYqUde --steps 3 --estimator exchange-classifier.joblib --height 716600 --maxexprank 2000 --output pony_exploration
```

The above command will generate a graph file in GML format. Once the graph is generated, you can search for relations within the graph using the lib/graph.py script:

```
python lib/graph.py --graph pony_exploration.gml --directed --decorate path/to/blocksci.cfg
```


## Clustering Heuristics

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

WatchYourBack will automatically create and load the clusters from such paths, transparent to the user. Be careful when using too many different heights, due to the disk space usage.


## Tags

In order to classify an address, WatchYourBack uses a set of tags which specify the owner of specifc addresses and their multi-input clusters. 
Identification of service addresses (e.g., exchanges) is fundamental to prevent the exploration from exploding, 
i.e., introducing many unrelated nodes in the exploration graph.

All tag files should be located in data/tagging/ folders, and must follow the next format:

- CSV file, no headers, no index, with six columns: address, ticker, category, tag, subtype, and url
- address, ticker, category and url are mandatory; tag and subtype can be empty


## Paths

The path search algorithm will look for either directed (--directed) or undirected (--undirected) paths between the entities in the graph. However, as stated in the [paper](https://doi.org/10.1145/3548606.3560587), meaningful relations are those that describe money flow, represented by directed paths. The argument --decorate can be used to print blockchain properties of addresses and transactions.


## Oracles

WatchYourBack can optionally make use of four classifiers (called oracles) to determine if an address found within the exploration can be linked to the following malware families: cerber, pony, skidmap, or glupteba.

Oracles can be invoked using --oracle [cerber|pony|glupteba]; Addresses detected by the oracle will be purple-colored in the resulting graph.


## Exchange-classifier

Addresses without tags could still belong to exchanges and thus introduce explosion in the exploration. 
WatchYourBack ships with a trained machine learning exchange classifier to determine if the address is possibly owned by an exchange.
Following are the instructions in case you need to re-train the ML exchange classifier.

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

## Citations

If you use this work, please use the next citations:

```
@inproceedings{10.1145/3548606.3560587,
author = {Gomez, Gibran and Moreno-Sanchez, Pedro and Caballero, Juan},
title = {Watch Your Back: Identifying Cybercrime Financial Relationships in Bitcoin through Back-and-Forth Exploration},
year = {2022},
isbn = {9781450394505},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3548606.3560587},
doi = {10.1145/3548606.3560587},
booktitle = {Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security},
pages = {1291–1305},
numpages = {15},
keywords = {blockchain, malware, cybercrime financial relations, clipper},
location = {Los Angeles, CA, USA},
series = {CCS '22}
}
```

```
@inproceedings{10.1145/3576915.3623094,
author = {Gomez, Gibran and van Liebergen, Kevin and Caballero, Juan},
title = {Cybercrime Bitcoin Revenue Estimations: Quantifying the Impact of Methodology and Coverage},
year = {2023},
isbn = {9798400700507},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3576915.3623094},
doi = {10.1145/3576915.3623094},
booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
pages = {3183–3197},
numpages = {15},
keywords = {cybercrime, revenue estimation, bitcoin, deadbolt ransomware},
location = {<conf-loc>, <city>Copenhagen</city>, <country>Denmark</country>, </conf-loc>},
series = {CCS '23}
}
```

