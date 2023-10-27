#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
import blocksci
import networkx as nx
from networkx.exception import NetworkXNoPath
from networkx.drawing.nx_agraph import graphviz_layout
from collections import defaultdict, OrderedDict
from datetime import datetime
from math import log
import time
import logging
try:
    from lib import tags, price_api as price
    from lib import blockchain_feature_extraction as bs_fs
except ModuleNotFoundError:
    import tags, price_api as price
    import blockchain_feature_extraction as bs_fs

chains = None
cms = None
utxos = defaultdict(set)
graph_cache = {}

def connect(pubkey, t1, a1, t2, a2):
    G = nx.DiGraph()
    attrs = {'type': 'pubkey', 'value': 0, 'trecv': 0,
            'tsent': 0, 'balance': 0}
    G.add_node(pubkey, attrs=attrs)
    G.add_edge(f"{t1}:{a1}", pubkey, weight=-1)
    G.add_edge(pubkey, f"{t1}:{a1}", weight=-1)
    G.add_edge(f"{t2}:{a2}", pubkey, weight=-1)
    G.add_edge(pubkey, f"{t2}:{a2}", weight=-1)
    return G

def addr_graph(saddr, addr, txes, height=0, prune=False):
    '''
    Generate a di-graph containing a single node for an address, and a node for
    each one of its deposit/withdrawal txes, connected by directed edges.
    Multiple edges to the same tx will result in an aggregation of weights.

    :param blockchain.Address addr: Address object for the address-node
    :param dict txes: Dict object with the keys 'd_txes' and 'w_txes'
    containing the list of deposit and withdrawal txes of type blocksci.Tx
    :param int height: Do not consider txes beyond specific blockchain height
    :param bool prune: Do not draw deposit txes if prune=True
    :return: The directed graph
    :rtype: nx.DiGraph
    '''
    G = nx.DiGraph()
    t_ = saddr.split(':')[0]
    trecv = sum(addr.outputs.value)
    tsent = sum(addr.inputs.value)
    balance = addr.balance(height) if height else addr.balance()
    node = {'type': 'addr', 'value': trecv, 'trecv': trecv,
            'tsent': tsent, 'cj': len(txes['coinjoin']), 'balance': balance}
    G.add_node(saddr, attrs=node)

    # Deposit txes
    for tx in txes['d_txes']:
        # Accumulate multiple deposits from the same addr
        value = sum([o.value for o in tx.outputs if o.address==addr])
        slots = [str(o.index) for o in tx.outputs if o.address==addr]
        ts = tx.block.timestamp
        index = tx.hash
        tname = f"{t_}:{index}"
        fdate = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
        btc_price = price.get_price(fdate)
        # Adding a weighted edge according to the BTC transfered
        usd = value * btc_price * 1e-8
        txnode = {'type': 'tx', 'ts': f"{datetime.utcfromtimestamp(ts)}",
                'in': tx.input_count, 'out': tx.output_count,
                'btcin': tx.input_value, 'btcout': tx.output_value,
                'value': tx.input_value - tx.fee, 'fee': tx.fee}
        G.add_node(tname, attrs=txnode)
        G.add_edge(tname, saddr, weight=value, usd=usd, slots=','.join(slots))
        # UTXOS
        if height:
            # TODO where filter
            addr_utxos = [o for o in addr.outputs.to_list() \
                    if o.spending_tx_index == None or \
                    o.spending_tx.block.height > height]
            # TODO This could be faster, but it has a bug right now
            # addr.outputs.unspent(height).to_list()
        else:
            addr_utxos = addr.outputs.unspent().to_list()
        for o in addr_utxos:
            utxos[saddr].add((tname, o.index, o.value))

    # Withdrawal txes
    for tx in txes['w_txes']:
        # Accumulate multiple withdrawals to the same addr
        value = sum([i.value for i in tx.inputs if i.address==addr])
        slots = [str(i.index) for i in tx.inputs if i.address==addr]
        ts = tx.block.timestamp
        index = tx.hash
        tname = f"{t_}:{index}"
        fdate = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
        btc_price = price.get_price(fdate)
        # Adding a weighted edge according to the BTC transfered
        usd = value * btc_price * 1e-8
        txnode = {'type': 'tx', 'ts': f"{datetime.utcfromtimestamp(ts)}",
                'value': value, 'in': tx.input_count, 'out': tx.output_count,
                'btcin': tx.input_value, 'btcout': tx.output_value,
                'fee': tx.fee}
        G.add_node(tname, attrs=txnode)
        G.add_edge(saddr, tname, weight=value, usd=usd, slots=','.join(slots))

    return G

def graph_compose_from_list(graphs, clear=False):
    '''
    Return one directed graph that is the composition of a list of graphs.

    :param list graphs: List of nx.DiGraph objects to join
    :param bool clear: Call clear() method for already added graphs
    :return: A graph composed from all graphs of the list
    :rtype: nx.DiGraph
    '''
    if not graphs:
        return None
    final_graph = graphs[0]
    for i in range(1, len(graphs)):
        final_graph = nx.compose(final_graph, graphs[i])
        if clear:
            graphs[i].clear()
    return final_graph

def draw_addr_graphs(g, addr_info, serv_txes, layout='graphviz', fn=''):
    '''
    Produce a gml file with the address graph given, with a specific layout.

    :param nx.DiGraph g: The graph to draw
    :param dict addr_info: Dict object with the info of each address specified
    in blocksci.Address:Address pairs.
    :param str layout: Use graphviz_layout to position nodes when using
    'graphviz' or 'twopi'. Use nx.spring_layout otherwise
    :param str fn: Name of the gml file
    :return: None
    '''
    # Use the addr name as key
    addrs = {v.fullname: v for a, v in addr_info.items()}
    if layout == 'graphviz':
        pos = graphviz_layout(g)
    elif layout == 'twopi':
        # Fix for twopi error for nodes with weight=0
        for e in g.edges:
            w = g.edges[e]['weight']
            g.edges[e]['weight'] = w if w else -1
        # -----
        pos = graphviz_layout(g, prog='twopi')
    else:
        pos = nx.spring_layout(g, k=0.7, iterations=20)

    logging.info(f"Saving {fn} in gml format")
    gformat = gml_format(g, pos, addrs, serv_txes)
    nx.readwrite.gml.write_gml(gformat, f"{fn}.gml")

def get_color(operation, tagged, cj, far, color):
    '''
    Get the color of a node based on flags.

    :param bool operation: The node belong to the operation
    :param bool tagged: The node has a tag
    :param bool cj: The node has CoinJoin txes
    :param bool far: The node has Force Address Reuse txes
    :param str color: The rest of the nodes use this two-bytes RGB color that
    depends on the distance to the seed nodes
    :return: a RGB color string
    :rtype: str
    '''
    if operation:
        color = "#7800ff" # Purple
    elif tagged:
        color = "#ff8000" # Orange
    elif cj or far:
        color = "#80ff00" # Light green
    else:
        color = f"#00{color}00" # Some green
    return color

def gml_format(G, pos, addr_info, serv_txes):
    '''
    Give GML format to a nx.DiGraph object by adding some fields to specify
    the position of the nodes, its color, its type, etc.

    :param nx.DiGraph G: The graph
    :param dict pos: A dict object containing node:position pairs for all nodes
    in the graph
    :param dict addr_info: Dict object with the info of each address specified
    in blocksci.Address:Address pairs.
    :return: The graph with GML columns
    :rtype: nx.DiGraph
    '''
    for x in G.nodes:
        G.nodes[x]['graphics'] = {}
        G.nodes[x]['graphics']['x'] = pos[x][0]
        G.nodes[x]['graphics']['y'] = pos[x][1]
        G.nodes[x]['graphics']['z'] = 0
        value = int(G.nodes[x]['attrs']['value'])
        G.nodes[x]['value'] = value
        if G.nodes[x]['attrs']['type'] == 'tx':
            ts = G.nodes[x]['attrs']['ts']
            label = f"{x} | {ts} | BTC: {value * 1e-8:.8f}"
            G.nodes[x]['type'] = 'tx'
            G.nodes[x]['Polygon'] = 4
            if x in serv_txes['serv']:
                G.nodes[x]['graphics']['fill'] = '#000088'
            elif x in serv_txes['multi']:
                G.nodes[x]['graphics']['fill'] = '#004c88'
            else:
                G.nodes[x]['graphics']['fill'] = '#0000ff'
        elif G.nodes[x]['attrs']['type'] == 'pubkey':
            label = f"{x}"
            G.nodes[x]['type'] = 'pubkey'
            G.nodes[x]['Polygon'] = 5
            G.nodes[x]['graphics']['fill'] = '#dcdcdc'
        else:
            G.nodes[x]['cluster'] = addr_info[x].cid
            G.nodes[x]['ctags'] = addr_info[x].ctags
            G.nodes[x]['owner'] = addr_info[x].owner
            G.nodes[x]['tag'] = addr_info[x].tag
            G.nodes[x]['fulltag'] = addr_info[x].fulltag
            G.nodes[x]['csize'] = addr_info[x].csize
            G.nodes[x]['service'] = addr_info[x].service
            G.nodes[x]['step'] = addr_info[x].step
            G.nodes[x]['seed'] = addr_info[x].seed
            G.nodes[x]['exprank'] = addr_info[x].txes['exp_rank']
            G.nodes[x]['steplabel'] = f"{addr_info[x].step}:{x}"
            G.nodes[x]['Polygon'] = 0
            G.nodes[x]['coinjoin'] = G.nodes[x]['attrs']['cj']
            if addr_info[x].blocklist:
                G.nodes[x]['type'] = 'blocklist'
                G.nodes[x]['graphics']['fill'] = '#000000'
                label = f'{x}'
            elif addr_info[x].exch:
                G.nodes[x]['type'] = 'exchange'
                G.nodes[x]['graphics']['fill'] = '#ff0000'
                G.nodes[x]['prediction'] = addr_info[x].pred
                label = f"[{addr_info[x].cid}] {x}"
            else:
                G.nodes[x]['type'] = 'addr'
                label = f'{x}'
                step = int(addr_info[x].step)
                operation = addr_info[x].isop()
                cj = len(addr_info[x].txes['coinjoin']) > 0
                far = len(addr_info[x].txes['far']) > 0
                # TODO Don't support more than 15 steps!
                color = hex(0xff - (0x10 * step)).replace('0x', '')
                trecv = G.nodes[x]['attrs']['trecv'] * 1e-8
                tsent = G.nodes[x]['attrs']['tsent'] * 1e-8
                label = f"s{step} | {x}:{addr_info[x].cid}"
                label += f" | Recv: {trecv:.8f} BTC"
                label += f" | Sent: {tsent:.8f} BTC"
                label += ' | COINJOIN' if cj else ''
                label += ' | FAR' if far else ''
                G.nodes[x]['prediction'] = addr_info[x].pred
                G.nodes[x]['operation'] = operation
                tagged = bool(addr_info[x].tag)
                color = get_color(operation, tagged, cj, far, color)
                G.nodes[x]['graphics']['fill'] = color
                # Change color and label to nodes with utxos
                if x in utxos:
#                    ul = ','.join([f"{u[0]}:{u[1]}" for u in sorted(utxos[x])])
#                    label += f" | UTXOS: {ul}"
                    ul = sum([u[2] for u in utxos[x]])
                    label += f" | UTXOS ({len(utxos[x])}): {ul/1e8:.8f} BTC"
                    b = G.nodes[x]['attrs']['balance']
                    label += f" | Balance: {b}"
                    # TODO do we want to color addrs with UTXOS?
#                    G.nodes[x]['graphics']['fill'] = '#ffff00'\
#                            if not operation else "#7800ff"
        G.nodes[x]['labelfull'] = label
        G.nodes[x]['attrs'] = {}
    return G

def operation(g, addr_info):
    """
    Search for additional addresses that belong to the operation by searching
    paths between two nodes of the operation or cycle paths containing at least
    one operation node. Additional operation nodes are added from the same
    clusters than the seeds, if any.
    The procedure should do the same after adding more clusters to the list.

    :param nx.DiGraph g: The graph
    :param dict addr_info: Dict object with the info of each address specified
    in blocksci.Address:Address pairs.
    :return: A tuple of (addr_info, clusters) with the updated addr_info dict
    and the set of cluster-ids found.
    """
    # Transform to address graph to calc paths
    G = address_graph(g, directed=True)

    # Exclude OP_RETURN addresses (they are dead ends for the path search)
    op_ret = {(t, a): v for (t, a), v in addr_info.items() \
            if a.type == blocksci.address_type(6)}
    # Use addr.fullname as key
    addrs = {v.fullname: v for (tckr, a), v in addr_info.items() \
            if a.type != blocksci.address_type(6)}

    # Search for the MI-clusters (op-clusters) of all our op-addrs
    percs = defaultdict(set)
    for fullname, addr in addrs.items():
        if addr.isop():
            percs[(addr.ticker, addr.cid)].add(addr.op.rate)
    clusters = set(percs.keys())
    logging.debug(f">>> Initial Clusters: {percs}")

    # Search for all addresses in our op-clusters
    op_addrs = set()
    for fullname, addr in addrs.items():
        ckey = (addr.ticker, addr.cid)
        if ckey in clusters:
            if addr.op is None:
                addr.update(op={'perc': max(percs[ckey]), 'iocs': []})
                op_addrs.add(fullname)
            else:
                op_addrs.add(fullname)
    logging.debug(f">>> INITIAL OP ADDRS: {op_addrs}")

    # All addr-nodes in the graph
    addr_nodes = set([x for x in G.nodes \
            if G.nodes[x]['attrs']['type'] == 'addr'])

    # Search for cycles with op nodes. If some op node is in the cycle, all the
    # list should be made of op nodes
    orig_op_addrs = op_addrs.copy()
    t1 = time.time()
    cycles = [n for n in nx.simple_cycles(G) if len(n) > 1]
    while cycles:
        new_op_addrs = set()
        cycles_del = set()
        for n, l in enumerate(cycles):
            if set(l).intersection(op_addrs):
                logging.debug(f">>> CYCLE PATH: {l}")
                cycles_del.add(n)
                new_op_addrs.update(set(l).intersection(addr_nodes))
        cycles = [l for n, l in enumerate(cycles) if n not in cycles_del]
        if not new_op_addrs:
            break
        op_addrs.update(new_op_addrs)

    # Update op-clusters
    # TODO We can include elements of op-clusters found in cycles
    clusters.update([(addrs[x].ticker, addrs[x].cid) for x in op_addrs \
            if x in addrs])
    logging.debug(f">>> CYCLE OP ADDRS: {op_addrs - orig_op_addrs}")
    logging.debug(f"Time in cycles: {time.time() - t1}")

    # Search for all addrs in the path from an op-node to each other
    new_op_addrs = set()
    paths = defaultdict(int)
    visited = set()
    t1 = time.time()
    # Search paths between every two nodes, originals op-nodes only
    # due to the search can not be acheived in polynomial time, we perform
    # it into the address-graph that is smaller.
    for x in orig_op_addrs:
        for l in nx.all_simple_paths(G, x, orig_op_addrs - {x}):
            new_op_addrs.update(l)
            paths[x] += 1
        logging.debug(f">>> PATHS FROM {x} TO all: {paths[x]}")
    new_op_addrs = new_op_addrs.intersection(addr_nodes)
    logging.debug(f">>> SIMPLE-PATH OP ADDRS: {new_op_addrs}")

    # Update op-clusters
    clusters.update([(addrs[x].ticker, addrs[x].cid) for x in new_op_addrs \
            if x in addrs])

    # Update the 'op' field of all addrs found
    n_op = new_op_addrs - op_addrs
    logging.debug(f">>> NEW OP ADDRS: {len(n_op)}\t{n_op}")
    logging.debug(f"Time in all simple paths: {time.time() - t1}")
    op_addrs.update(n_op)
    for x in op_addrs:
        if addrs[x].op is None:
            addrs[x].update(op={'perc': 1.0, 'iocs': []})

    # Use addr.addr as key
    addrs = {(v.ticker, v.addr): v for a, v in addrs.items()}
    # Add OP_RETURN addresses
    addrs.update(op_ret)

    return addrs, clusters

def explosion_node_rank(G, height, threshold=0):
    '''
    Return an OrderedDict(src_node: {set of dst nodes}) where the set
    corresponds to the dst nodes that would be disconnected if the src node
    were removed from the graph. The threshold can be used to remove src nodes
    from the ranking with less or equal that number of dst nodes.

    :param nx.DiGraph G: The graph
    :param int threshold: The minimum number of dst nodes from each ranked node
    :return: An OrderedDict with { src_node: {set of distinct nodes} } pairs
    :rtype: OrderedDict
    '''
    tagged = tagged_nodes(G, height)
    if len(tagged) < 2:
        logging.info(f"Not enough tagged nodes: {tagged}")
        return {}

    logging.info("Computing explosion rank (undirected graph)")
    g = G.to_undirected()
    # TODO is it better to calculate the explosion on address graphs?
#    g = address_graph(G, directed=False)


    # Get node connectivity information from graph
    H = nx.algorithms.connectivity.build_auxiliary_node_connectivity(g)
    R = nx.algorithms.flow.build_residual_network(H, 'capacity')

    # Update destination map with nodes in disjoing paths between
    # src nodes to dst nodes
    seen = set()
    path_nodes = defaultdict(set)
    for src in tagged:
        seen.add(src)
        for dst in set(tagged) - seen: #nodes_dst - nodes_src:

            # cutoff=2 to limit the number of paths found, we'll ignore'em
            paths, num_paths = disjoint_paths(g, src, dst, 2, H, R)
            logging.debug(f"Found {num_paths} path(s) between {src} and {dst}")
            logging.debug('\n'.join([':'.join(p) for p in paths]))

            # Ignore pairs without paths or with multiple paths
            if not num_paths or num_paths > 1:
                continue

            # Update map of nodes: set()
            for path in paths:
                for node in path[1:-1]:
                    path_nodes[node].add(dst)

    # Sort nodes by number of destinations and degree
    sorted_pairs = sorted(path_nodes.items(), reverse=True,
            key=lambda x: (len(x[1]), g.degree[x[0]]))

    # Produce ranking removing overlaps and applying threshold
    out_map = OrderedDict()
    prev_set = set()
    for (node, dst) in sorted_pairs:
        num_all_dst = len(dst)
        # If threshold not met, we can finish
        if num_all_dst <= threshold:
            break
        # Substract previous set to remove overlaps
        diff_set = dst.difference(prev_set)
        num_dst = len(diff_set)
        logging.info(f"{node}: {num_all_dst} all dst, {len(diff_set)} diff set")
        # Check threshold
        if num_dst > threshold:
            out_map[node] = diff_set
            # Update prev_set
            prev_set.update(diff_set)

    return out_map

def components(G, fname=''):
    '''
    Return the list of connected components in a graph.

    :param nx.DiGraph G: The graph
    :return: The list of connected components
    :rtype: list
    '''
    # for undirected graphs only
    comp = list(nx.connected_components(G.to_undirected()))
    logging.info(f"Components in the graph: {len(comp)}")
    seeds = {x for x in G.nodes if 'seed' in G.nodes[x] and G.nodes[x]['seed']}
    tagged = {x for x in G.nodes \
            if ('fulltag' in G.nodes[x] and G.nodes[x]['fulltag']) \
                or ('owner' in G.nodes[x] and G.nodes[x]['owner']) \
                or ('ctags' in G.nodes[x] and G.nodes[x]['ctags'])}
    jsonf = []
    for n, c in enumerate(sorted(comp, key=len, reverse=True)):
        logging.info(f"Component {n}:")
        logging.info(f"\tNodes: {len(c)}")
        s = seeds & c
        logging.info(f"\tSeeds: {len(s)}")
        jseeds = []
        jtags = {}
        for x in sorted(s):
            logging.info(f"\t\t{x}")
            jseeds.append(x)
        t = {x: G.nodes[x]['fulltag'] for x in (tagged & c)}
        logging.info(f"\tTagged nodes: {len(t)}")
        for x, xt in sorted(t.items(), key=lambda n: n[1]):
            logging.info(f"\t\t{x} {xt}")
            jtags.update({x: xt})
        j = {
                'component': n,
                'nodes': len(c),
                'nseeds': len(s),
                'ntagged': len(t),
                'seeds': jseeds,
                'tagged': jtags
            }
        jsonf.append(j)
    with open(fname, 'w') as f:
        f.write('\n'.join([json.dumps(j) for j in jsonf]))
    return comp

def tagged_nodes(G, height, split=False, services=False):
    '''
    If split=False return a dict with all tagged nodes. If split=True, return
    a tuple (tagged, seeds, target), with a dict with all tagged nodes, a set
    of all seed nodes, and a set of all service nodes found from tagged nodes.

    :param nx.DiGraph G: The graph
    :param bool split: Specify that a cashout tuple should be returned
    :return: tagged nodes, and optionally a set of seed nodes and service nodes
    for the cashout path search.
    '''
    tagged = {x: G.nodes[x]['fulltag'] for x in G.nodes \
            if 'fulltag' in G.nodes[x] and G.nodes[x]['fulltag']}
    logging.info(f"Found {len(tagged)} tagged addresses in the graph")
    if split | services:
        logging.info(f"Searching for seed/service nodes")
        cseeds = {G.nodes[x]['cluster'] for x in G.nodes \
                if 'seed' in G.nodes[x] and G.nodes[x]['seed']}
        seeds = {x for x in tagged if G.nodes[x]['cluster'] in cseeds}
        # in case the seeds are not tagged:
        if not seeds:
            seeds = {x for x in G.nodes if 'cluster' in G.nodes[x] and G.nodes[x]['cluster'] in cseeds}
            tagged.update({x: 'seed=seed' for x in seeds})
        if services:
            target = {x for x in G.nodes if 'ctags' in G.nodes[x] \
                    and tags.is_service_ctags(G.nodes[x]['ctags'])}
        else:
            target = {x for x in G.nodes if 'ctags' in G.nodes[x] \
                    and G.nodes[x]['ctags'] and x not in seeds}

        logging.info(f"Found {len(cseeds)} seed clusters: {cseeds}")
        logging.info(f"Found {len(seeds)} seeds: {seeds}")
        logging.info(f"Targeting {len(target)} tagged{' services' if services else ''} addresses")
        return tagged, seeds, target
    return tagged

def decorate(G, blockscip, ticker, paths, tagged, height=0):
    '''
    Decorate paths by adding cluster-id and tags to address nodes and number of
    input/output slots to txes

    :param G: networkx Graph
    :param blockscip: BlockSci config file to work with
    :param ticker: Ticker of the Blockchain to work with
    :param dict paths: Dict with the paths to decorate in the form
    {(src, dst): ([list of paths], n_paths)}
    :param dict tagged: Pairs of {address: tag}
    :param int height: Height of the blockchain to initialize blocksci data
    :return: Dict with the new list of decorated paths
    :rtype: dict
    '''
    init_blocksci(blockscip, ticker, height=height)
    global chains
    global cms
    dpaths = {}
    for k, v in paths.items():
        # TODO if k[0] or k[1] are not in tagged should be exchange-classified
        # addresses, should we use the prediction value instead?
        src_tag = f":{tagged[k[0]]}" if k[0] in tagged else ':EC'
        dst_tag = f":{tagged[k[1]]}" if k[1] in tagged else ':EC'
        src = f"{k[0]}:{G.nodes[k[0]]['cluster']}{src_tag}"
        dst = f"{k[1]}:{G.nodes[k[1]]['cluster']}{dst_tag}"
        pair = src, dst
        pair_paths = []
        for p in v[0]:
            path = []
            dpath = ''
            addr = None
            tx = None
            pk = None
            for i, e in enumerate(p):
                addr_to_tx = False
                if i%2 == 0: # address
                    (t_, a) = e.split(':')
                    addr = bs_fs.addr_from_string(a, chains[t_], height)
                    node = f"{e}:{G.nodes[e]['cluster']}"
                    node += f":{tagged[e]}" if e in tagged else ''
                else: # transaction or pubkey
                    try:
                        (t_, t) = e.split(':')
                        tx = chains[t_].tx_with_hash(t)
                        node = f"{e}({tx.input_count}|{tx.output_count})"
                        addr_to_tx = True
                        pk = None
                    except:
                        tx = None
                        node = f"{e}"
                        pk = e
                path.append(node)
                dir = ''
                if tx and addr:
                    # TODO add edge[usd] to the edges
                    if addr in tx.outputs.address.to_list():
                        dir += ' <- ' if addr_to_tx else ' -> '
                    if addr in tx.inputs.address.to_list():
                        dir += ' -> ' if addr_to_tx else ' <- '
                    dir = ' <-> ' if dir == ' ->  <- ' else dir
                    dir = ' <-> ' if dir == ' <-  -> ' else dir
                elif addr and pk:
                    dir = ' <-> '
                dpath += dir + node
            pair_paths.append(dpath)
        dpaths[pair] = pair_paths, len(pair_paths)
    return dpaths

def all_disjoint_paths(G, height, directed, cashouts=False, deposits=False,
        ignore_ec=False):
    '''
    Search all disjoint paths (directed or undirected) between sets of nodes:
    from seeds to services if cashouts=True, or from services to seeds if
    deposits=True, or from any tagged node to each other otherwise. Then, split
    paths containing multiple tagged nodes to avoid duplicate paths. Exchange-
    classified addresses count as tagged when splitting paths, unless
    ignore_ec=True.

    :param nx.DiGraph G: The graph
    :param bool directed: Search for directed paths if True, or undirected
    paths otherwise
    :param bool cashouts: Search for cashout payments (from seeds to services)
    :param bool deosits: Search for deposit payments (from services to seeds)
    :param bool ignore_ec: If True, ignore exchange-classified addresses when
    splitting paths with multiple tags
    :return: A tuple (tagged, paths) with the tagged nodes found in the graph
    and the paths in the form {(src, dst): ([list of paths], n_paths)}
    :rtype: tuple
    '''
    tagged, seeds, target = tagged_nodes(G, height=height, split=True, services=cashouts|deposits)
    if len(tagged) < 2:
        logging.info(f"Not enough tagged nodes: {tagged}")
        return {}, {}, {}

    G = G if directed else G.to_undirected()
    H = nx.algorithms.connectivity.build_auxiliary_node_connectivity(G)
    R = nx.algorithms.flow.build_residual_network(H, 'capacity')
    paths = {}

    # Search paths
    if cashouts | deposits:
        # between seeds and services
        services = target
        # d-paths from seeds to services if cashouts, or vice versa if deposits
        src, dst = (seeds, services) if cashouts else (services, seeds)
        src_s, dst_s, m = ('seeds', 'services', 'cashout') if cashouts \
                else ('services', 'seeds', 'deposit')
        m = f"Searching {m} paths from {len(src)} {src_s} to {len(dst)} {dst_s}"
        logging.info(m)
        for x in src:
            for y in dst:
                paths[(x, y)] = disjoint_paths(G, x, y, cutoff=None, H=H, R=R)
    elif directed:
        # between all tagged nodes, both ways due A->B != B->A
        logging.info(f"Searching directed paths from seeds to all tagged nodes")
        for x in seeds:
            for y in target:
                paths[(x, y)] = disjoint_paths(G, x, y, cutoff=None, H=H, R=R)
        for x in target:
            for y in seeds:
                paths[(x, y)] = disjoint_paths(G, x, y, cutoff=None, H=H, R=R)
    else:
        seen = set()
        # between all tagged nodes, just once due A->B == B->A
        logging.info(f"Searching undirected paths between all tagged nodes")
        for x in tagged:
            seen.add(x)
            for y in set(tagged) - seen:
                paths[(x, y)] = disjoint_paths(G, x, y, cutoff=None, H=H, R=R)

    # Split paths having more than one tag
    exch = {} if ignore_ec else \
            {x for x in G.nodes if G.nodes[x]['type'] == 'exchange'}
    logging.info(f"Loaded {len(exch)} exchange-classified addresses")
    partial_paths = defaultdict(set)
    for pair, all_paths in paths.items():
        for path in all_paths[0]:
            start = pair[0]
            subpath = [start]
            for i, node in enumerate(path):
                if not i:
                    continue
                subpath.append(node)
                # subpath found
                # we also split a path when a classified-exchange addr is found
                if node in tagged or node in exch:
                    partial_paths[(start, node)].add('|'.join(subpath))
                    start = node
                    subpath = [start]

    paths = {pair: [p.split('|') for p in u_paths] \
            for pair, u_paths in partial_paths.items()}
    paths = {pair: (path, len(path)) \
            for pair, path in sorted(paths.items(), key=lambda x: x[0])}
    return tagged, seeds, paths

def disjoint_paths(g, src, dst, cutoff=None, H=None, R=None):
    '''
    Search for disjoint paths between two nodes in a graph

    :param nx.DiGraph g: The graph
    :param str src: Name of the source node
    :param str dst: Name of the destination node
    :param int cutoff: Maximum number of paths to yield
    :param nx.DiGraph H: Auxiliary digraph to compute flow based node
    connectivity
    :param nx.DiGraph R: Residual network to compute maximum flow
    :return: A tuple ([list of paths], n_paths) with the list of paths and the
    number of paths found
    :rtype: tuple
    '''
    try:
        # Get disjoint paths between two nodes
        paths = list(nx.node_disjoint_paths(g, src, dst, cutoff=cutoff,
            auxiliary=H, residual=R))
    # Ignore not connected nodes
    except nx.exception.NetworkXNoPath:
        logging.debug(f"No path between {src} and {dst}")
        return [], 0

    num_paths = len(paths)
    return paths, num_paths

def address_graph(G, gml=False, directed=True):
    '''
    Transform a transaction-address graph into an address graph by replacing
    all edges between pairs of (address, transaction) and the respective
    transaction nodes into edges between pairs of (address, address)

    :param nx.DiGraph G: The transaction graph
    :param bool gml: The graph comes from a GML file
    :param bool directed: Work with a directed graph if True, or an undirected
    graph otherwise
    :return: An address graph
    :rtype: nx.DiGraph
    '''
    if directed:
        Ga = nx.DiGraph()
    else:
        G = G.to_undirected()
        Ga = nx.Graph()
    nodes = [x for x in G.nodes if G.nodes[x]['type'] != 'tx'] if gml \
        else [x for x in G.nodes if G.nodes[x]['attrs']['type'] != 'tx']
    edges = set()
    for x in nodes:
        for xedge in G.edges(x):
            tx = xedge[1]
            edges.update([(x, e[1]) for e in G.edges(tx) if e[1] != x])
            # TODO add weights if needed
#            edges.update([(x, e[1]) for e in G.edges(tx, data=True) if e[1] != x])
    Ga.add_edges_from(edges)
    for x in Ga.nodes():
        Ga.nodes[x]['attrs'] = G.nodes[x]['attrs']
    return Ga

def init_blocksci(blockscip, ticker, height=0):
    '''
    Initialize global blocksci objects for reading blockchain data

    :param blockscip: Path to the blocksci config file
    :param height: Maximum height of the blockchain to parse
    :return: True if the initialization is successful, False otherwise
    :rtype: bool
    '''
    # Load the blockchains
    global chains
    global cms

#    chain, cm = bs_fs.build_load_blocksci(blockscip, height)
#    chains = {ticker: chain}
#    cms = {ticker: cm}

    # TODO Crosschain config
    data_dir = '/data/BlockSci'
    chains = {
            'btc': blocksci.Blockchain(f"{data_dir}/btc.cfg"),
            #'bch': blocksci.Blockchain(f"{data_dir}_Bitcoincash/bch.cfg"),
            'ltc': blocksci.Blockchain(f"{data_dir}_Litecoin/ltc.cfg")
        }

    return all(chains)

def print_table(g):
    comp = len(list(nx.connected_components(g.to_undirected())))
    addr = len([n for n in g.nodes if g.nodes[n]['type'] != 'tx'])
    tx = len([n for n in g.nodes if g.nodes[n]['type'] == 'tx'])
    unexp = len([n for n in g.nodes if g.nodes[n]['type'] == 'blocklist'])
    seeds = len([n for n in g.nodes \
            if 'seed' in g.nodes[n] and g.nodes[n]['seed'] == 1])
    texch = len([n for n in g.nodes if 'service' in g.nodes[n] and
        g.nodes[n]['service'] == 'exchange'])
    exch = len([n for n in g.nodes if g.nodes[n]['type'] == 'exchange']) - texch
    tagged = len([n for n in g.nodes if 'fulltag' in g.nodes[n] and
        g.nodes[n]['fulltag'] != ''])
    row = f"{seeds:,} & ? & ? & {comp:,} & {addr:,} "
    row += f"& {tx:,} & {unexp:,} & {tagged:,} & {exch:,}"
    print(row)

def prune_graph(G):
    '''
    Prune all nodes beyond any exchange-classified address node. Pruned nodes
    were included due to some addresses individually classified as non-exchange
    that are latter classified as exchanges due to it's MI-cluster.

    :param graph: A nx.DiGraph object
    :return: A new nx.DiGraph object already pruned
    :rtype: nx.DiGraph
    '''
    new = nx.DiGraph()
    visited = set()
    # Initialize the queue with all the seeds
    visit = {n: G.nodes[n] for n in G.nodes \
            if 'seed' in G.nodes[n] and G.nodes[n]['seed'] == 1}
    while(visit):
        node, attrs = visit.popitem()
        new.add_node(node, **attrs)
        visited.add(node)

        # Get input/output edges
        ins = G.in_edges(node, data=True)
        outs = G.out_edges(node, data=True)

        # Add all nodes and edges connected to the visited node
        new.add_edges_from(ins)
        new.add_nodes_from([(src, G.nodes[src]) for (src, dst, attrs) in ins])
        new.add_edges_from(outs)
        new.add_nodes_from([(dst, G.nodes[dst]) for (src, dst, attrs) in outs])

        # We don't want to visit exchange-classified nodes
        ins_n = {src: G.nodes[src] for (src, dst, attrs) in ins if not \
                    ('type' in G.nodes[src] and \
                        (G.nodes[src]['type'] == 'exchange' or \
                        G.nodes[src]['type'] == 'blocklist')
                    )}
        outs_n = {dst: G.nodes[dst] for (src, dst, attrs) in outs if not \
                    ('type' in G.nodes[dst] and \
                        (G.nodes[dst]['type'] == 'exchange' or \
                        G.nodes[dst]['type'] == 'blocklist')
                    )}
        visit.update({n: attrs for n, attrs in ins_n.items() \
                if n not in visited})
        visit.update({n: attrs for n, attrs in outs_n.items() \
                if n not in visited})
    return new

def get_serv(owner):
    serv = 'EC' if owner == 'EC' else tags.is_service_owner(owner)
    return serv if serv else 'Not a service'

def main(args):
    if args.table:
        print_table(args.graph)
        return

    if args.prune:
        G = prune_graph(args.graph)
        nx.readwrite.gml.write_gml(G, f"{args.prune}.gml")
        return

    if args.compose:
        ngraphs = len(args.compose)
        graph_list = []
        logging.info(f"Composing {ngraphs} graphs")
        for g in args.compose:
            try:
                graph_list.append(nx.readwrite.gml.read_gml(g))
            except e:
                logging.error(e)
                logging.error(f"Error reading {g}. Exiting.")
                return
        G = graph_compose_from_list(graph_list, clear=True)
        ofile = os.path.join(args.output, f"{ngraphs}_graphs.gml")
        nx.readwrite.gml.write_gml(G, ofile)
        return

    if args.directed or args.undirected:
        tagged, seeds, paths = all_disjoint_paths(args.graph, args.height,
                args.directed, args.cashouts, args.deposits, args.ignore_ec)
        paths = decorate(args.graph, args.decorate, args.ticker, paths, tagged,
                  args.height) if args.decorate else paths
        s = '>' if args.directed else '<>'
        services = defaultdict(set)
        slist = defaultdict(set)
        for k, v in paths.items():
            if v[1]:
                logging.info(f"Paths between {k[0]} and {k[1]}: {v[1]}")
                if args.decorate:
                    logging.info('\n'.join(v[0]))
                    src_t, src_addr, src_cid, src_ctag = k[0].split(':')
                    dst_t, dst_addr, dst_cid, dst_ctag = k[1].split(':')
                    src = ':'.join([src_t, src_addr])
                    dst = ':'.join([dst_t, dst_addr])
                    if src in seeds:
                        dtag = ':'.join([dst_cid, dst_ctag])
                        serv_name = dst_ctag.split('>>')[0]
                        serv_class = get_serv(serv_name)
                        serv = ':'.join(serv_name.split('='))
                        slist['To'].add(serv)
                        services[('To', serv_class)].update([dtag])
                    elif dst in seeds:
                        stag = ':'.join([src_cid, src_ctag])
                        serv_name = src_ctag.split('>>')[0]
                        serv_class = get_serv(serv_name)
                        serv = ':'.join(serv_name.split('='))
                        slist['From'].add(serv)
                        services[('From', serv_class)].update([stag])
                else:
                    logging.info('\n'.join([s.join(p) for p in v[0]]))
        if args.decorate:
            logging.info('Summary by class, entity and cluster:')
            for (d, service), entities in sorted(services.items()):
                logging.info(f"{d} Service == {service}: {len(entities)}")
                logging.info('\n'.join(sorted(entities)))
            logging.info('General summary:')
            for d, l in slist.items():
                l.discard('')
                logging.info(f"{d}: {', '.join(sorted(l))}")
    elif args.rank:
        out_map = explosion_node_rank(args.graph, threshold=0)
        logging.info(out_map)
    elif args.paths or args.upaths:
        d = bool(args.paths)
        src, dst = args.paths.split('|') if d else args.upaths.split('|')
        G = args.graph if d else args.graph.to_undirected()
        s = '>' if d else '<>'
        paths = disjoint_paths(G, src, dst)
        if paths[1]:
            logging.info(f"Paths between {src} and {dst}: {paths[1]}")
            if args.decorate:
                tagged = tagged_nodes(G, args.height)
                path = decorate(G, args.decorate, args.ticker,
                        {(src, dst): paths}, tagged, args.height)
                logging.info('\n'.join(path.popitem()[1][0]))
            else:
                path = paths[0]
                logging.info('\n'.join([s.join(p) for p in path]))
    elif args.components:
        components(args.graph, args.components)

if __name__ == '__main__':
    version = "2.4.3"
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', action='store', type=str,
            help='Graph in GML format')
    parser.add_argument('-K', '--ticker', dest='ticker', default='btc',
            choices=['btc', 'bch', 'ltc'], help='Blockchain to work with')
    parser.add_argument('-d', '--directed', action='store_true',
            help='Search all directed paths between tagged nodes')
    parser.add_argument('-u', '--undirected', action='store_true',
            help='Search all undirected paths between tagged nodes')
    parser.add_argument('-c', '--cashouts', action='store_true',
            help='Search possible cash-out paths between seeds and services')
    parser.add_argument('-r', '--deposits', action='store_true',
            help='Search possible deposit paths between services and seeds')
    parser.add_argument('-R', '--rank', action='store_true',
            help='Calculate explosion rank between tagged nodes')
    parser.add_argument('-p', '--paths', action='store', type=str,
            help='Search directed disjoint paths between src:dst nodes')
    parser.add_argument('-P', '--upaths', action='store', type=str,
            help='Search undirected disjoint paths between src:dst nodes')
    parser.add_argument('-C', '--components', action='store_true',
            help='Output connected components information of this graph')
    parser.add_argument('-D', '--decorate', action='store', default=None,
            help='BlockSci config file, used to load blockchain data used to \
                    decorate paths with address tags and txes size (ins/outs)')
    parser.add_argument('-i', '--ignore-ec', action='store_true',
            help='Ignore exchange-classified addresses in path search')
    parser.add_argument('-t', '--table', action='store_true',
            help='Print table values')
    parser.add_argument('-X', '--prune', action='store', type=str,
            help='Prune graph by cutting nodes beyond exchange-classified \
            addresses')
    parser.add_argument('-H', '--height', action='store', default=0, type=int,
            help='Height for the cluster tags')
    parser.add_argument('-O', '--output', action='store', type=str,
            help='Name for the output folder', default='')
    parser.add_argument('-G', '--compose', action='append', default=None,
            help='Compose several graphs in GML format into a single file')
    parser.add_argument('-v', '--version', action='version', version=version)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Made with version {version}")
    logging.debug(f"- Added colors for service and multiservice txes")
    logging.debug(f"- Added exp_rank to the graph")
    logging.debug(f"- Added --compose parameter")
    logging.debug(f"- Use tx.hash instead of tx.index")
    logging.debug(f"- Added full tags (ctag>>tag)")
    logging.debug(f"- Added tx-slots to graph edges")
    logging.debug(f"- Updated summary")
    logging.debug(f"- Fix initial addresses and rates in path search")
    logging.debug(f"- Changed directed path search (Services summary)")
    logging.debug(f"- Removed OP_RETURN addrs from path search")
    logging.debug(f"- Implemented op-search with cross-chain")
    logging.debug(f"- Implemented path search with cross-chain")
    logging.debug(f"- Added ticker to txes")
    logging.debug(f"- Added ticker parameter")
    logging.debug(f"- Added 'summary' of services found for cashouts/deposits")
    logging.debug(f"- Added 'mining' tag as a service")
    logging.debug(f"- Fixed: search for all tagged nodes for cashouts/deposits")
    logging.debug(f"- Exchange-classified nodes are considered tagged when")
    logging.debug(f"    splitting paths with tags>1 unless using --ignore-ec")
    logging.debug(f"{args}")

    if args.compose and not os.path.isdir(args.compose[0]):
        for g in args.compose:
            if not os.path.isfile(g):
                e = 'File {g} not found'
                parser.error(e)
    elif args.compose and os.path.isdir(args.compose[0]):
        e = f"Graph files not found in {args.compose[0]}"
        args.compose = [g for g in glob.glob(f"{args.compose[0]}/*.gml")]
        if not args.compose:
            parser.error(e)
    elif not args.graph or not os.path.isfile(args.graph):
        e = 'Graph file not found.'
        parser.error(e)
    else:
        if args.components:
            args.components = args.graph.replace('.gml', '_components.jsonl')
        args.graph = nx.readwrite.gml.read_gml(args.graph)

    if args.decorate and not os.path.isfile(args.decorate):
        e = "Blocksci data not found."
        parser.error(e)

    if (args.paths and len(args.paths.split('|')) != 2) or \
            (args.upaths and len(args.upaths.split('|')) != 2):
        e = '[U]Path search parameter should be src|dst'
        parser.error(e)

    main(args)

