#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import time
import signal
import random
import logging
import blocksci
import argparse
import pandas as pd
from lib import blockchain_feature_extraction as bfe
from lib import malware, tags, graph
from lib.address import Address
from collections import defaultdict
from joblib import load

class Exploration:
    ''' Main class to handle explorations '''

    # Set default values for control objects
    def __init__(self, ticker, blocksci, height, estimator, seeds,
            inputfile, expand, oracle, threshold, steps, max_nodes, epsilon,
            skip_back, skip_first_back, skip_forth, maxexprank, output, paths,
            exchanges):
        # Set exploration parameters
        self.ticker = ticker
        self.blocksci = blocksci
        self.height = height
        self.expand = expand
        self.steps = steps
        self.oracle = oracle
        self.paths = paths
        self.threshold = threshold
        self.max_nodes = max_nodes
        self.maxexprank = maxexprank
        self.skip_back = skip_back
        self.skip_first_back = skip_first_back
        self.skip_forth = skip_forth
        self.epsilon = epsilon
        self.output = output

        # Load blockchain data, seeds and estimator
        self.load_blockchain()
        self.load_seeds(seeds, inputfile)
        self.load_estimator(estimator)
        self.load_glupteba_keys()

        # Control variables for the exploration
        self.addr_graphs = {}
        self.clusters = {}
        self.exchanges = {}
        self.expanded = defaultdict(set)
        self.verified = set()
        self.operation = set()
        self.addresses = {}
        self.predictions = {}
        self.seen = set()
        self.seed_clusters = set()
        self.seed_tags = set()
        self.visited_txes = {'btc': set(), 'bch': set(), 'ltc': set()}
        self.visit_later = {}
        self.non_visited = {}
        self.blocklist = set()
        self.directions = defaultdict(set)
        self.addrtxes = {}
        self.addr_info = {}
        self.cross_join = defaultdict(set)
        self.opclusters = set()
        self.serv_txes = {'serv': set(), 'multi': set()}
        self.step = 0

    def load_glupteba_keys(self):
        if not (self.oracle and self.oracle == 'glupteba'):
            self.glup_keys = None
            return
        # TODO read these from args?
        k1 = 'd8727a0e9da3e98b2e4e14ce5a6cf33ef26c6231562a3393ca465629d66503cf'
        k2 = '1bd83f6ed9bb578502bfbb70dd150d286716e38f7eb293152a554460e9223536'
        self.glup_keys = [k1, k2]

    def load_blockchain(self):
        """ Load blockchain and multi-input data from blocksci """
        # Load the blockchain
        chain, cm = bfe.build_load_blocksci(self.blocksci, self.height)
        self.chains = {self.ticker: chain}
        self.cms = {self.ticker: cm}

        n = len(self.chains)
        logging.debug(f"Loaded {n} blockchain{'s' if n > 1 else ''}.")
        for k, v in self.chains.items():
            csize = self.cms[k].clusters().size
            logging.debug(f"{k.upper()} height: {v[-1].height:,}")
            logging.debug(f"{k.upper()} clusters: {csize:,}")


    def load_seeds(self, seeds, inputfile):
        lseeds = {}
        if seeds:
            for tseed in seeds:
                t, seed = (self.ticker, tseed)
                # TODO height just for BTC for now
                h = self.height if t == 'btc' else 0
                seed, txes = bfe.addr_in_out_txes_from_string(seed,
                        self.chains[t], h)
                if not seed:
                    continue
                lseeds[(t, seed)] = txes
        elif inputfile:
            for a, d in bfe.load_seeds(inputfile).items():
                t = self.ticker
                # TODO height just for BTC for now
                h = self.height if t == 'btc' else 0
                seed, txes = bfe.addr_in_out_txes_from_string(a,
                        self.chains[t], h)
                if not seed:
                    continue
                lseeds[(t, seed)] = txes

        if not lseeds:
            logging.error('No seed could be loaded. Exiting.')
            exit(1)

        for (t, seed), txes in lseeds.items():
            logging.info(f"Loaded seed {seed} with {txes['total']} txes")

        self.seeds = lseeds


    def load_estimator(self, estimator):
        """ Load an exchange classifier """
        if estimator:
            self.estimator = load(estimator)
        else:
            self.estimator = None
            logging.warning('Exchange classifier not set.')


    def build_seeds(self):
        for (t, addr), txes in self.seeds.items():
            self.build_addr(addr, txes, ticker=t)


    def build_addr(self, addr, txes, cid=None, cowner=None, ctags=None,
            csize=None, exch=False, blocklist=False, ticker=None):
        """
        Build the context of a Bitcoin address. If the cluster it belongs to is
        tagged as a service it won't be explored. Services are:
        exchange/onlinewallet/mixer/tormarket/mining/payment-processor/service
        """
        t = ticker if ticker is not None else self.ticker
        h = self.height if t == 'btc' else 0
        name = bfe.addr_to_string(addr)
        fname = f"{t}:{name}"
        owner, tag, _ = tags.search_tag(name, t, h)
        key = (t, addr)
        ckey = (t, cid)
        service = ''
        # Get information about the cluster
        if cid is None:
            cm = self.cms[t]
            cluster = cm.cluster_with_address(addr)
            cid = cluster.index
            ckey = (t, cid)
            rt = tags.search_tag_by_cluster(cid=cid, ticker=t, height=h)
            cowner, ctags, csize = rt
            msg = f"{fname}: Building cluster info cid:{cid}"
            msg += f" cowner:{cowner} ctags: {ctags} height:{h}"
            logging.debug(msg)

            # We generate tags for BitMEX addresses on the fly
            addnew = False
            if name[:5].lower() == '3bmex':
                addnew = bfe.bitmex_classifier(addr, self.chains[t])
                if addnew:
                    cowner = 'exchange=bitmex'
                    tag = 'exchange=bitmex=bmex-classifier'
                    nt = f"{t}={tag}"
                    ctags = ';'.join(ctags.split(';')+[nt]) if ctags else nt
                    csize = cluster.type_equiv_size

            # Check if the cluster has a service tag or if it is an
            # exchange-classified (or BitMEX) cluster
            service, exch = self.register_exchange(addr, name, cid, ctags, t,
                    addnew)

        is_seed = key in self.seeds
        if is_seed:
            logging.info(f"{fname} is a seed, cid:{ckey} added to seeds")
            self.seed_clusters.add(ckey)
            self.seed_tags.add(tag)

        step = self.step
        address = Address(addr, txes, cid=cid, owner=cowner, tag=tag,
                ctags=ctags, csize=csize, step=step, seed=is_seed, exch=exch,
                blocklist=blocklist, service=service, ticker=t)
        logging.info(f"Built address {name} fulltag:{address.fulltag}")

        # Determine if this addr belongs to the operation
        if (not service and not exch) or is_seed:
            self.apply_oracle(address)
        # Save the address
        self.addr_info[key] = address

        return address


    def register_exchange(self, addr, name, cid, ctags, tckr, addnew=False):
        key = (tckr, addr)
        ckey = (tckr, cid)
        fname = f"{tckr}:{name}"
        service = tags.is_service_ctags(ctags)
        cserv = 'exchange' == service or 'onlinewallet' == service
        cexcl = ckey in self.clusters and self.clusters[ckey]
        if not addnew and not (cserv or cexcl):
            return service, False
        elif cserv:
            msg = f"{fname}: Cluster {cid} is tagged as exchange due to "
            msg += f"ctags {ctags}. Service {service}"
        else:
            msg = f"{fname}: Address belongs to an exchange cluster: {cid}"
        logging.info(msg)

        if self.expand:
            if cexcl:
                self.clusters[ckey].add(addr)
                # cid should exist in 'exchanges' with all its addresses unless
                # it has a service tag
                if cserv:
                    s = next(iter(self.exchanges[ckey]))
                    self.exchanges[ckey][s].add(key)
            else:
                self.clusters[ckey] = {addr}
                self.exchanges.update({ckey: {name: {key}}})
        else:
            if cexcl:
                s = next(iter(self.exchanges[ckey]))
                self.exchanges[ckey][s].add(key)
            else:
                self.clusters[ckey] = True
                self.exchanges.update({ckey: {name: {key}}})
        return service, True


    def apply_oracle(self, address):
        # Apply oracle on addrs found
        addr = address.addr
        txes = address.txes
        op = None
        if self.oracle == 'cerber':
            op = malware.oracle_cerber(addr, txes['w_txes'])
        elif self.oracle == 'pony':
            op = malware.oracle_pony(addr, txes)
        elif self.oracle == 'glupteba':
            op = malware.oracle_glupteba(txes['w_txes'], self.glup_keys)
        elif self.oracle == 'deadbolt':
            op = malware.oracle_deadbolt(txes['w_txes'])

        if op and op['perc'] >= 0.5:
            address.update(op=op)
            self.opclusters.add(address.cid)
            self.operation.add(address)
            msg = f"{address.fullname}: Operation address found: {address.op}"
            logging.warning(msg)


    def expand_verify(self, addrs):
        new_nodes = {}
        for (tckr, a), txes in addrs.items():
            if (tckr, a) in self.verified:
                continue
            self.verified.add((tckr, a))

            cid, caddrs = self.mi_expand(a, txes, tckr=tckr)
            # TODO Do we still want to explore online-wallets for seeds?
            if not caddrs and self.addr_info[(tckr, a)].seed:
                caddrs = {(tckr, a): txes}
                msg = f"{tckr}:{a}: Exploring seed in exchange cluster"
                logging.warning(msg)
            if caddrs:
                self.expanded[(tckr, cid)].update(caddrs.keys())
            new_nodes.update(caddrs)

            self.verified.update(caddrs)

        return new_nodes


    def mi_expand(self, seed, txes, tckr='btc'):
        """ Return all addresses in the cluster where seed is found """
        address = self.addr_info[(tckr, seed)]
        fname = address.fullname
        cid = address.cid
        cluster = self.cms[tckr].clusters()[cid]
        ckey = (tckr, cid)

        if self.expand:
            logging.info(f"{fname}: Expanding addr of type {seed.type}")

            if (tckr, cid) in self.clusters:
                logging.info(f"\tCluster {tckr}:{cid} is already included")
                self.clusters[(tckr, cid)].add(seed)
                return cid, {}

            caddrs = set(cluster.addresses.to_list())
            # TODO some of these could be expensive
            logging.info(f"\tRaw address: {seed}")
            logging.info(f"\tAs input in {seed.input_txes_count()} TXES")
            logging.info(f"\tAs output in {seed.output_txes_count()} TXES")
            logging.info(f"\tEquiv addresses {seed.equiv().addresses.size}")
            logging.info(f"Cluster: {cid}")
            logging.info(f"\tSize {len(caddrs)} (TE size: {cluster.type_equiv_size})")
            self.clusters[(tckr, cid)] = set()
            tp, caddrs = self.detect_exchange(address, caddrs)
            if tp:
                self.clusters[(tckr, cid)].add(seed)
                return cid, {}
            else:
                return cid, caddrs
        else:
            if ckey in self.clusters and self.clusters[ckey]:
                # and not address.seed
                msg = f"{fname}: Address belongs to an "
                msg += f"exchange cluster: {cid}"
                logging.info(msg)
                # cid should exist in exchanges
                s = next(iter(self.exchanges[ckey]))
                self.exchanges[ckey][s].add((tckr, seed))
                # Add exch addr with txes already in the exploration
                address.update(exch=True)
                return cid, {}

            tp = self.detect_address_exchange(address)
            if tp:
                self.clusters[ckey] = True
                self.update_exchanges(seed, cid, ticker=tckr)
                return cid, {}
            else:
                return cid, {(tckr, seed): txes}


    def detect_exchange(self, address, caddrs):
        seed = address.addr
        stxes = address.txes
        tckr = address.ticker
        cid = address.cid
        cowner = address.cowner
        ctags = address.ctags
        csize = address.csize
        serv_tag = 'exchange' if address.exch else address.service
        saddr = address.name
        name = address.fullname

        # Support for no estimator
        if not self.estimator:
            return False, caddrs

        valid_types = {x: None for x in self.estimator['preprocessor'].\
                named_transformers_['cat']['onehot'].categories[0]}

        # TODO keep the original seed, check if txes are missed due equiv addr
        caddrs.discard(address.addr)
        caddrs = bfe.addresses_txes(caddrs, self.height, filter_coinjoin=True,
                filter_far=True)
        # Build other addresses of the same cluster
        caddrs_tmp = {}
        caddresses = {}
        while caddrs:
            addr, txes = caddrs.popitem()
            caddrs_tmp[(tckr, addr)] = txes
            caddresses[a.name] = self.build_addr(addr, txes, cid, cowner,
                    ctags, csize, ticker=tckr)
        # Add the seed to the set of addresses
        caddrs_tmp[(tckr, seed)] = stxes
        caddresses[address.name] = address
        caddrs = caddrs_tmp

        # Do not classify operation addresses
        if cid in self.opclusters:
            return False, caddrs

        # don't make exchange predictions on seeds or MI of seeds
        if (tckr, cid) in self.seed_clusters:
            return False, caddrs

        # don't make exchange predictions on exchange-tagged clusters
        if serv_tag:
            msg = f"{name}: Exchange classifier skips this serv-address "
            msg += f"{serv_tag}"
            logging.debug(msg)
            if serv_tag == 'exchange':
                # TODO check if we need this update here
                self.exchanges.update({(tckr, cid): {saddr: caddrs.keys()}})
                return True, caddrs
            else:
                return False, caddrs

        # We sort the addresses by number of txes, ascending
        prior = sorted(caddrs.items(), key=lambda x: x[1]['total'])
        exchange_detected = False
        timer_init = time.time()
        produce_vector = bfe.produce_vector
        addr_to_string = bfe.addr_to_string
        dataframe = pd.DataFrame
        predict_proba = self.estimator.predict_proba
        batch_size = 10
        vectors = []
        predictions = {}
        # TODO vectors can be produced in batches too, it's faster
        # vectors, t = produce_vectors(caddrs, chain, cm, args.height, 'unknown')
        for n, ((t_, addr), txes) in enumerate(prior):

            if exchange_detected:
                break

            if 'bfork_txes' in txes:
                txes = txes['bfork_txes']
            else:
                txes = txes['txes'] | txes['coinjoin'] | txes['far']

            eaddr = addr_to_string(addr)
            v, t = produce_vector(addr, self.chains[t_], self.cms[t_],
                    txes=txes, height=self.height if t_=='btc' else 0)
            vectors.append([v for k, v in v.items()])

            # TODO remove label from vectors...
            #vectors.append([v for k, v in v.items() if k != 'label'])
            msg = f"Time elapsed extracting features for {t_}:{eaddr}: {t}"
            logging.debug(msg)
            try:
                valid_types[v['type']]
            except KeyError:
                logging.debug(f">> Unseen address type: {v['type']}")

            if len(vectors) == batch_size or n == len(prior)-1:
                df = dataframe(data=vectors, columns=v.keys())
                X = df.drop(['label'], axis=1)
                # TODO ... to do something like (remove label from keys):
                #X = dataframe(data=vectors, columns=list(v.keys())[1:])
                # seed_classes = predict(X)
                seed_classes = predict_proba(X)
                for idx, label in enumerate(seed_classes):
                    # Save prediction results
                    # estimator.classes_ == array(['exchange', 'non-exchange']
                    # i = int(n/batch_size)*batch_size + idx
                    predictions[X.iloc[idx].address] = label[0]
                    if label[0] >= self.threshold:
                        msg = f">> Cluster {t_}:{cid} classified as exchange"
                        logging.info(msg)
                        logging.info(f">> Seed: {seed}")
                        logging.info(f">> Vector {n}: {X.iloc[idx]}")
                        d = {(t_, cid): {saddr: caddrs.keys()}}
                        self.exchanges.update(d)
                        exchange_detected = True
                        break
                vectors = []

        # Update predictions
        for a, p in predictions.items():
            caddresses[a].update(pred=p, exch=exchange_detected)
        self.predictions.update({(tckr, k): v for k, v in predictions.items()})

        timer_end = time.time() - timer_init
        logging.debug(f"Time elapsed on cluster {tckr}:{cid}: {timer_end}")
        return exchange_detected, caddrs


    def detect_address_exchange(self, address):
        addr = address.addr
        txes = address.txes
        cid = address.cid
        tckr = address.ticker
        serv_tag = 'exchange' if address.exch else address.service
        saddr = address.name
        name = address.fullname

        # Support for no estimator
        if not self.estimator:
            return False

        valid_types = {x: None for x in self.estimator['preprocessor'].\
                named_transformers_['cat']['onehot'].categories[0]}
        if not txes:
            msg = f"detect_address_exchange querying for txes: {name}"
            logging.warning(msg)
            txes = bfe.address_in_out_txes(addr, self.height,
                    filter_coinjoin=True, filter_far=True)

        # Do not classify operation addresses
        if address.cid in self.opclusters:
            logging.debug(f"{name}: Exchange classifier skips an op-address")
            return False

        # don't make exchange predictions on seeds or MI of seeds
        if (tckr, cid) in self.seed_clusters:
            logging.debug(f"{name}: Exchange classifier skips a seed address")
            return False

        # don't make exchange predictions on tagged clusters
        if serv_tag:
            msg = f"{name}: Exchange classifier skips a serv-addr {serv_tag}"
            logging.debug(msg)
            if serv_tag == 'exchange':
                # TODO check if we need this update here
                self.exchanges.update({(tckr, cid): {saddr: {(tckr, addr)}}})
                return True
            else:
                return False

        exchange_detected = False
        timer_init = time.time()

        if 'bfork_txes' in txes:
            txes = txes['bfork_txes']
        else:
            txes = txes['txes'] | txes['coinjoin'] | txes['far']

        v, t = bfe.produce_vector(addr, self.chains[tckr], self.cms[tckr],
                txes=txes, height=self.height if tckr=='btc' else 0)
        logging.debug(f"Time elapsed extracting features for {name}: {t}")
        try:
            valid_types[v['type']]
        except KeyError:
            logging.debug(f">> Unseen address type: {v['type']}")

        X = pd.DataFrame(data=[v]).drop(['label'], axis=1)
        seed_classes = self.estimator.predict_proba(X)

        # =====================================================================
        # Introduce errors with probability of Epsilon=x between 20
        # e.g. in 1 of 20 cases (5%) we flip the results
        if self.epsilon:
            epsilon = self.epsilon.split(':')
            epsilon_num, epsilon_stop = int(epsilon[0]), int(epsilon[1])
            # FN when result >= th; FP when result < th
            if (random.randint(1, epsilon_stop) <= epsilon_num) and \
                    seed_classes[0][0] < self.threshold:
                logging.debug(f"Flipping the classifier results !!!")
                seed_classes[0][0] = 1 - seed_classes[0][0]
        # =====================================================================

        # Save prediction results
        # estimator.classes_ == array(['exchange', 'non-exchange']
        self.predictions[(tckr, saddr)] = seed_classes[0][0]
        address.update(pred=seed_classes[0][0])
        if seed_classes[0][0] >= self.threshold:
            msg = f">> Addr {name} from cluster {cid} classified as exchange"
            logging.info(msg)
            logging.info(f">> Vector: {X.iloc[0]}")
            self.exchanges.update({(tckr, cid): {saddr: {(tckr, addr)}}})
            exchange_detected = True
            address.update(exch=True)
        timer_end = time.time() - timer_init
        logging.debug(f"Time elapsed on address {name}: {timer_end}")

        return exchange_detected


    def update_exchanges(self, seed, cid, ticker):
        """ Search for nodes of the same cluster that are included in other
        steps of the trace, or in the same step, to remove them.
        """
        # TODO: If undetected exchange addresses were included previously, some
        # of their txes could introduce addresses transacting with the exchange
        # that are not an exchange or that are not related with the operation,
        # even if we remove all the addresses of that cluster here
        saddr = bfe.addr_to_string(seed)
        for s in range(len(self.addresses)):
            if (ticker, cid) in self.addresses[s]:
                caddrs = self.addresses[s].pop((ticker, cid))
                self.exchanges[(ticker, cid)][saddr].update(caddrs)
                for (t, ca) in caddrs:
                    self.addr_info[(t, ca)].update(exch=True)
                msg = f"Elements of cluster {t}:{cid} found at step "
                msg += f"{s}: {caddrs}"
                logging.info(msg)
        if (ticker, cid) in self.expanded:
            caddrs = self.expanded.pop((ticker, cid))
            self.exchanges[(ticker, cid)][saddr].update(caddrs)
            for (t, ca) in caddrs:
                self.addr_info[(t, ca)].update(exch=True)
            msg = f"Elements of cluster {t}:{cid} found in current step:"
            msg += f" {caddrs}"
            logging.info(msg)


    def explore(self):
        global stop_signal

        # Expand seeds
        self.visit_later = self.expand_verify(self.seeds)
        logging.debug(f"Expanded seeds: {self.visit_later.keys()}")
        self.seen.update(self.visit_later.keys())

        # Save expanded addresses into the current step
        self.addresses[self.step] = self.expanded

        for (t_, addr) in self.visit_later:
            self.directions[(t_, addr)] = set()

        # Variable to control the max_nodes argument
        total_nodes = 0
        # Exploration loop
        while self.visit_later and self.step < self.steps and not stop_signal:
            self.step += 1
            logging.info(f"========= Step {self.step} ==========")

            # Worklist sorted by desc priority to pop lower exp_rank elements
            self.non_visited = {k: v for k, v in sorted(self.visit_later.items(),
                key=lambda x: x[1]['exp_rank'], reverse=True)}
            logging.info(f"Non visited nodes: {len(self.non_visited)}")

            # Init control variables
            self.visit_later = {}
            self.expanded = defaultdict(set)
            timer_init = time.time()

            # Begin to visit addrs
            while self.non_visited:
                (t_, node), txes = self.non_visited.popitem()
                address = self.addr_info[(t_, node)]
                fname = address.fullname

                # Do not explore exchange-classified or service addresses
                if (address.exch or address.service) and not address.seed:
                    reason = 'service-tag' if address.service \
                            else 'exchange-classified'
                    logging.info(f"{fname}: Node not visited ({reason})")
                    continue

                # Stop the exploration if self.max_nodes is reached
                stop_signal = self.verify_stop_max_nodes()
                if stop_signal:
                    break

                logging.info(f"Visiting: {fname} Exp-rank: {txes['exp_rank']}")
                dd_a, dw_a, wd_a, ww_a = self.visit_node(node, txes, t_)

                # Build graph of the current addr
                h = self.height if t_ == 'btc' else 0
                # TODO The graph can be pruned using prune=True
                g = graph.addr_graph(fname, node, txes, height=h, prune=False)
                self.addr_graphs[fname] = g
                nnodes, nedges = len(g.nodes), len(g.edges)
                msg = f"{fname}: Graph nodes: {nnodes} edges: {nedges}"
                logging.debug(msg)

                # Stop the exploration if self.max_nodes is reached
                stop_signal = self.verify_stop_max_nodes()
                if stop_signal:
                    break

                # Expand addresses found, and add'em to the worklist
                self.visit_later.update(self.expand_verify(dd_a))
                self.visit_later.update(self.expand_verify(dw_a))
                self.visit_later.update(self.expand_verify(wd_a))
                self.visit_later.update(self.expand_verify(ww_a))
                # TODO Should we add all new nodes found to the seen set? When
                # using expand=True some addrs could be explored twice?

                if stop_signal:
                    msg = "stop_signal received, unexplored addresses left: "
                    msg += f"{len(self.non_visited)}"
                    logging.warning(msg)
                    break

            self.addresses[self.step] = self.expanded
            timer_end = time.time() - timer_init
            logging.info(f"Time elapsed on step {self.step}: {timer_end}")

        logging.info(f"Exploration stopped at step {self.step}")
        logging.info(f"Nodes seen: {len(self.addr_info)}")
        logging.info(f"Non visited nodes: {len(self.non_visited)}")
        logging.info(f"Visite later nodes: {len(self.visit_later)}")

        self.remaining_nodes()

        logging.info(f'{len(self.addr_graphs)} graphs generated.')
        logging.info(f'{len(self.addr_info)} addresses seen.')

        list_graphs = list(self.addr_graphs.values())

        return graph.graph_compose_from_list(list_graphs, True)


    def visit_node(self, node, node_txes, ticker):
        """
        Search for deposit/withdrawal txes for a given address, omiting already
        visited txes. We will search for addresses depositing/withdrawing from
        those txes, to add them to the work list.
        """
        # We want to visit unvisited txes only
        w_txes = node_txes['w_txes'] - self.visited_txes[ticker]
        self.visited_txes[ticker].update(node_txes['w_txes'])
        d_txes = node_txes['d_txes'] - self.visited_txes[ticker]
        self.visited_txes[ticker].update(node_txes['d_txes'])

        key = (ticker, node)
        ckey = (ticker, self.addr_info[key].cid)

        # Deposit and Withdrawal addresses of deposit txes
        # Omit deposit txes if skip_back is set, or if skip_first_back is set and
        # the cluster of such address is in a seed cluster
        if self.skip_back or \
            (self.skip_first_back and (ckey in self.seed_clusters)):
            dd_a, dw_a = {}, {}
        else:
            dd_a, dw_a = self.found_addrs_from_txes(node, d_txes, 'B', ticker)

        # Deposit and Withdrawal addresses of withdrawal txes
        # Omit withdrawal txes if skip_forth or if it's a service-owned seed
        # (e.g. a seed in an online-wallet)
        if self.skip_forth or \
            (self.addr_info[key].seed and self.addr_info[key].service):
            wd_a, ww_a = {}, {}
        else:
            wd_a, ww_a = self.found_addrs_from_txes(node, w_txes, 'F', ticker)

        return dd_a, dw_a, wd_a, ww_a


    def found_addrs_from_txes(self, node, txes, d, tckr):
        """
        Search for deposit/withdrawal addresses in the set of txes. We don't
        want to explore backwards if args.skip-back is set, so we omit deposit
        txes. We don't want to explore forwards if arg.skip-forth is set, so
        we omit withdrawal txes. We evaluate every addr found to update the
        work list.
        """
        d_found = defaultdict(dict)
        w_found = defaultdict(dict)
        # Sort txes by index so we can repeat the same exploration
        for tx in sorted(txes, key=lambda x:x.index):

            # Inputs, depositing
            # Omit deposit addrs if skip_back
            if self.skip_back:
                d_addrs = set()
            else:
                all_da = {(tckr, a) for a in tx.inputs.address.to_list()}
                d_addrs = all_da - self.seen
                self.seen.update(d_addrs)

            # Outputs, withdrawing
            # Omit withdrawal addrs if skip_forth
            if self.skip_forth:
                w_addrs = set()
            else:
                all_wa = {(tckr, a) for a in tx.outputs.address.to_list()}
                w_addrs = all_wa - self.seen
                self.seen.update(w_addrs)

            d_found.update(self.add_or_blocklist(node, d_addrs, d, 'd'))
            w_found.update(self.add_or_blocklist(node, w_addrs, d, 'w'))

        logging.info(f"{tckr}:{node}: Found {len(d_found)} {d}d addresses")
        logging.info(f"{tckr}:{node}: Found {len(w_found)} {d}w addresses")

        return d_found, w_found


    def add_or_blocklist(self, node, addrs, d, dd):
        '''
        Every new address found is build for a set of txes, cid, tag, ctags
        (if any) and explosion rank (maxexprank). If the addr has a service tag
        we don't want to explore it. If args.maxexprank is set, we blocklist
        all nodes with an explosion rank higher than args.maxexprank, to don't
        explore them. The set of nodes to reach the address is updated in
        `directions` object.
        '''
        found = defaultdict(dict)
        # Sort addrs by index so we can repeat the same exploration
        for (tckr, a) in sorted(addrs, key=lambda x: x[1].address_num):

            # Stop the exploration if args.max_nodes is setted
            if self.max_nodes and len(self.addr_info) > self.max_nodes:
                msg = f"Maximum number of nodes reached (while visiting) "
                msg += f"({len(self.addr_info)} of {self.max_nodes})"
                logging.warning(msg)
                break

            a_h = self.height if tckr == 'btc' else 0
            a_txes = bfe.address_in_out_txes(a, height=a_h)

            address = self.build_addr(a, a_txes, ticker=tckr)
            # Check if this addr has a service tag
            if address.service:
                msg = f"Skipping addr {tckr}:{address.name} of cluster "
                msg += f"{address.cid} due to service-flag: {address.ctags}"
                logging.info(msg)
                continue

            # Skip large addrs if maxexprank is set
            if self.maxexprank and a_txes['exp_rank'] > self.maxexprank:
                msg = f"Skipping address {tckr}:{a} with exp_rank of "
                msg += f"{a_txes['exp_rank']}. Found as {d}{dd} of {node}."
                logging.info(msg)
                address.update(blocklist=True)
                self.blocklist.add(address)
            else:
                found[(tckr, a)].update(a_txes)
                r = next(iter(self.directions[(tckr, node)])) \
                        if self.directions[(tckr, node)] \
                        else bfe.addr_to_string(node)
                self.directions[(tckr, a)].add(f"{r}:{d}{dd}:{address.name}")

        return found


    def remaining_nodes(self):
        """ Build graphs for exchange, blocklisted and visit_later nodes """
        for (t_, addr), a in self.addr_info.items():
            if a.fullname not in self.addr_graphs:
                # Trim txes of this address, we want to include explored only
                if a.exch or a.service or a.blocklist:
                    a.visited_txes(self.visited_txes[t_])
                    # remove nodes without visited txes
                    a_tx = a.txes['d_txes'] or a.txes['w_txes']
                    if not a_tx:
                        msg = f"Removed node {a.fulname} without visited txes"
                        logging.debug(msg)
                        continue
                    msg = 'Service' if a.exch or a.service else 'Blacklisted'
                    # Blacklisted addresses in exchange-clusters are exchanges
                    ckey = (t_, a.cid)
                    if a.blocklist and ckey in self.clusters \
                            and self.clusters[ckey]:
                        a.update(exch=True, blocklist=False)
                        # TODO do we need to update 'clusters' if expand=True?
                        s = next(iter(self.exchanges[ckey]))
                        self.exchanges[ckey][s].add((t_, addr))
                else:
                    msg = 'visit_later'
                a_h = self.height if t_ == 'btc' else 0
                g = graph.addr_graph(a.fullname, a.addr, a.txes, height=a_h)
                self.addr_graphs[a.fullname] = g
                nn, ne = len(g.nodes), len(g.edges)
                msg = f"{a.fullname} {msg} Graph nodes: {nn} edges: {ne}"
                logging.debug(msg)


    def path_search(self, g):
        """
        Mark addresses that possibly belong to the same operation by performing
        a directed path search through all operation nodes.
        """
        self.addr_info, self.clusters = graph.operation(g, self.addr_info)


    def draw(self, g):
        """ Draw and save the graph in GML format """
        l = 'twopi'
        graph.draw_addr_graphs(g, self.addr_info, self.serv_txes, layout=l, fn=self.output)


    def save_files(self):
        self.save_nonvisited()
        self.save_opaddresses()
        self.save_tags_found()
        self.save_iocs()
        self.save_exchanges()
        self.save_exchanges_reached()
        self.save_blocklisted()


    def save_nonvisited(self):
        # Save non_visited nodes
        if stop_signal and self.non_visited:
            with open(f"{self.output}.nvn", 'w') as f:
                for (t_, addr), txes in self.non_visited.items():
                    saddr = bfe.addr_to_string(addr)
                    f.write(f"{t_}:{saddr}\t{txes['total']}\n")


    def save_opaddresses(self):
        # Save to a file all addresses that could be part of the op
        with open(f"{self.output}.tsv", 'w') as fo:
            for step, exp_seeds in self.addresses.items():
                for (t_, cid), addrs in exp_seeds.items():
                    for (t__, addr) in addrs:
                        address = self.addr_info[(t__, addr)]
                        name = address.fullname
                        cid = address.cid
                        p = address.pred
                        # -1    -1: addr not in operation
                        # -1.0  False: oracle returned -1.0 (no w_txes)
                        #  x    False/True: oracle returned x
                        if self.oracle and address.op:
                            o = f"{address.op.rate}\t{address.isop()}"
                        else:
                            o = "-1\t-1"
                        d = '|'.join(sorted(self.directions[(t_, addr)]))
                        t = address.txes['total']
                        fo.write(f"{name}\t{cid}\t{step}\t{t}\t{d}\t{p}\t{o}\n")


    def save_tags_found(self):
        # Save tags found in the whole exploration
        with open(f"{self.output}.tag", 'w') as ft:
            for k, address in self.addr_info.items():
                if address.fulltag:
                    ft.write(f"{address.fullname}\t{address.fulltag}\n")


    def save_iocs(self):
        """
        Save information about the IOCs found for each operation address.
        """
        if not self.operation or not self.oracle:
            return

        rev_iocs = defaultdict(set)
        with open(f"{self.output}.ioc", 'w') as fo:
            for address in sorted(self.operation, key=lambda x: x.fullname):
                name = address.fullname
                for e in address.op.iocs:
                    if self.oracle in ['cerber', 'pony']:
                        tx1, tx2 = e['tx1'], e['tx2']
                        ts1, ts2 = e['ts1'], e['ts2']
                        v = e['domain' if self.oracle == 'cerber' else 'ip']
                        fo.write(f"{name}\t{tx1}\t{tx2}\t{ts1}\t{ts2}\t{v}\n")
                    elif self.oracle == 'glupteba':
                        key = e['key']
                        tx, ts, oidx = e['tx'], e['ts'], e['oidx']
                        v = e['domain']
                        fo.write(f"{name}\t{key}\t{tx}\t{oidx}\t{ts}\t{v}\n")
                    elif self.oracle == 'deadbolt':
                        ransom_addr = e['ransom_addr']
                        v = e['op_ret']
                        fo.write(f"{name}\t{ransom_addr}\t{v}\n")
                    elif self.oracle == 'bitab':
                        abuse_id = e['abuse_type_id']
                        abuse_ot = e['abuse_type_other']
                        abuser = e['abuser']
                        v = f"{abuse_id}_{abuse_ot}_{abuser}"
                        fo.write(f"{name}\t{abuse_id}\t{abuse_ot}\t{abuser}\n")

                    rev_iocs[v].add(address.fullname)

        # Log info about re-using of IOCs
        for v, addrs in rev_iocs.items():
            if len(addrs) > 1:
                na = len(addrs)
                logging.warning(f"IOC {v} found for {na} addresses: {addrs}")


    def save_exchanges(self):
        # Save possible exchanges
        with open(f"{args.output}.xch", 'w') as fo:
            for (t_, cid), e in self.exchanges.items():
                for seed, addrs in e.items():
                    for (t__, addr) in addrs:
                        address = self.addr_info[(t__, addr)]
                        name = address.fullname
                        cid = address.cid
                        p = address.pred
                        fo.write(f"{t__}:{cid}\t{seed}\t{name}\t{p}\n")


    def save_exchanges_reached(self):
        # Save contacted exchange addresses when using MI-expanding
        if self.expand:
            with open(f"{args.output}.cea", 'w') as fo:
                for (t_, cid), addrs in self.clusters.items():
                    for addr in addrs:
                        fo.write(f"{t_}:{bfe.addr_to_string(addr)}\t{cid}\n")


    def save_blocklisted(self):
        # Save blocklisted addresses
        with open(f"{self.output}.blk", 'w') as fo:
            for address in sorted(self.blocklist, key=lambda x: x.fullname):
                row = f"{address.fullname}\t{address.cid}"
                row += f"\t{address.txes['exp_rank']}\n"
                fo.write(row)


    def verify_stop_max_nodes(self):
        if self.max_nodes and len(self.addr_info) > self.max_nodes:
            msg = f"Maximum number of nodes reached (after visiting)"
            msg += f" ({len(self.addr_info)} of {self.max_nodes})"
            msg += f" unexplored addresses left: {len(self.non_visited)}"
            logging.warning(msg)

            return True

        return False



stop_signal = False


def clean_exit(sig, frame):
    global stop_signal
    logging.warning(f"[!]")
    logging.warning(f"[!] Received stop signal. Stopping uncompleted exploration.")
    logging.warning(f"[!]")
    stop_signal = True


def main(args):
    global stop_signal

    # Setting function handler for clean exit
    signal.signal(signal.SIGINT, clean_exit)

    logging.info(f"Loading blockchain with BlockSci v{blocksci.VERSION}")
    logging.info(f"BlockSci config file: {args.blocksci}")

    exploration = Exploration(**vars(args))

    # Get context information about seeds
    exploration.build_seeds()

    # Explore
    final_graph = exploration.explore()

    logging.info(f'Composed graph has {len(final_graph.nodes)} nodes.')

    # Search paths between operation nodes
    if args.paths and not stop_signal:
        exploration.path_search(final_graph)

    exploration.draw(final_graph)

    exploration.save_files()
    logging.info(f'Done.')


if __name__ == '__main__':
    usage = "\n\n\tExplore the blockchain transactions associated to an \
            initial set of seeds, both backwards and forward, then repeat the \
            exploration recursively at each step. Service addresses are not \
            explored. Produce an address-transaction graph at the end."

    version = "2.1.4"

    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('-D', '--blocksci', dest='blocksci', action='store',
            type=str, help='Blocksci config file')
    parser.add_argument('-s', '--seeds', action='append', type=str,
            help='Seed addresses to explore')
    parser.add_argument('-i', '--inputfile', dest='inputfile', action='store',
            type=str, help='File having seed addresses to explore')
    parser.add_argument('-S', '--steps', dest='steps', action='store',
            type=int, default=1, help='Max in/out transactions to trace')
    parser.add_argument('-H', '--height', dest='height', action='store',
            type=int, default=0, help='Max block height to expand')
    parser.add_argument('-e', '--estimator', dest='estimator', action='store',
            type=str, default='', help='Path to the exchange classifier')
    parser.add_argument('-o', '--output', dest='output', action='store',
            type=str, default='expanded', help='Name of the output file')
    parser.add_argument('-t', '--threshold', dest='threshold', action='store',
            type=float, default=0.5, help='Threshold to decide between classes')
    parser.add_argument('-T', '--maxexprank', dest='maxexprank', action='store',
            type=int, default=0, help='Discard tracking addreses with an\
                     explosion rank larger than this number')
    parser.add_argument('-b', '--skip-first-backwards', dest='skip_first_back',
            action='store_true', help='Skip first step backwards',
            default=False)
    parser.add_argument('-B', '--skip-back', dest='skip_back',
            action='store_true', help='Skip exploring backwards',
            default=False)
    parser.add_argument('-F', '--skip-forth', dest='skip_forth',
            action='store_true', help='Skip exploring forwards',
            default=False)
    parser.add_argument('-E', '--expand', dest='expand', default=False,
            action='store_true', help='Use multi-input clustering to expand \
                    found nodes when tracing transactions')
    parser.add_argument('-O', '--oracle', dest='oracle', default=None,
            choices=['cerber', 'pony', 'glupteba', 'deadbolt'],
            help='Set the oracle to identify operation addresses')
    parser.add_argument('-p', '--paths', dest='paths', default=False,
            action='store_true', help='Search for paths between op nodes')
    parser.add_argument('-X', '--exchanges', dest='exchanges', action='store',
            type=str, default=None, help='Exchanges identified in previous runs')
    parser.add_argument('-k', '--epsilon', dest='epsilon', action='store',
            type=str, default=None, help='Epsilon with format num:stop')
    parser.add_argument('-K', '--ticker', dest='ticker', default='btc',
            choices=['btc', 'bch', 'ltc'], help='Blockchain to work with')
    parser.add_argument('-N', '--max-nodes', dest='max_nodes', action='store',
            type=int, default=None, help='Max number of nodes to explore')
    parser.add_argument('-v', '--version', action='version', version=version)

    args = parser.parse_args()

    if not (args.blocksci and os.path.isfile(args.blocksci)):
        e = "Blocksci data not found."
        parser.error(e)
    if not (args.seeds or args.inputfile):
        e = "Seeds not found, use either --seeds or --inputfile to load seeds."
        parser.error(e)
    if args.inputfile and not os.path.isfile(args.inputfile):
        e = "Input file does not exist."
        parser.error(e)
    if args.estimator and not os.path.isfile(args.estimator):
        e = "Estimator file does not exist."
        parser.error(e)
    if args.exchanges and not os.path.isfile(args.exchanges):
        e = "Exchanges file don't exist."
        parser.error(e)

    logging.basicConfig(filename=f"{args.output}.log", level=logging.DEBUG)
    logging.debug(f"Made with version {version}")
    logging.debug(f"{args}")

    # Init the random seed to reproduce results when using epsilon
    if args.epsilon:
        random.seed(13)

    main(args)

