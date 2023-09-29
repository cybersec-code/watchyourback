#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import logging
import blocksci
import argparse
import pandas as pd
from collections import defaultdict
from datetime import datetime
try:
    from lib import tags
    from lib.Vector import Vector
except ModuleNotFoundError:
    import tags
    from Vector import Vector


bmex_addrs = set()
cm = None
chain = None

def build_art_clusters(chain, mapfile, height=0, ignore_wd=False):
    '''
    Use a mapfile to produce artificial clusters in dict format. Each cluster
    will have four fields: saddrs (bitcoin address in string format), addrs
    (bitcoin address in blocksci format), output_txes and input_txes (all
    output/input txes associated to the set of bitcoin addresses that belong to
    this cid).
    The mapfile should be in csv format, it should contain the headers and two
    columns: cid and addr (e.g. 'cid,addr')
    '''
    clusters = defaultdict(list)
    d = pd.read_csv(mapfile)
    for i, r in d.iterrows():
        clusters[r.cid].append(r.addr)
#        if r.cid not in clusters:
#            clusters[r.cid] = {
#                        'saddrs': set(),
#                        'addrs': set(),
#                        'output_txes': set(),
#                        'input_txes': set(),
#                    }
#        a = addr_from_string(r.addr, chain)
#        if not a:
#            continue
#        clusters[r.cid]['saddrs'].add(r.addr)
#        clusters[r.cid]['addrs'].add(a)
#        clusters[r.cid]['output_txes'].update(a.output_txes.to_list())
#        clusters[r.cid]['input_txes'].update(a.input_txes.to_list())
#    return clusters
    return build_art_clusters_from_dict(chain, clusters, height, ignore_wd)

def build_art_clusters_from_dict(chain, dictmap, height=0, ignore_wd=False):
    '''
    Use a dict to produce artificial clusters in dict format. Each cluster
    will have four fields: saddrs (bitcoin address in string format), addrs
    (bitcoin address in blocksci format), output_txes and input_txes (all
    output/input txes associated to the set of bitcoin addresses that belong to
    this cid). If a height is given, only txes up to this height will be
    returned.
    The mapfile should be in csv format, it should contain the headers and two
    columns: cid and addr (e.g. 'cid,addr')
    '''
    clusters = {}
    for cid, addrs in dictmap.items():
        if cid not in clusters:
            clusters[cid] = {
                        'saddrs': set(),
                        'addrs': set(),
                        'output_txes': set(),
                        'input_txes': set(),
                    }
        for addr in addrs:
            if isinstance(addr, str):
                saddr = addr
                a = addr_from_string(addr, chain, height)
            else:
                saddr = addr_to_string(addr)
                a = addr
            if not a:
                continue
            if height:
                o_txes = a.output_txes.where(lambda t: t.block.height <= height).to_list()
                if ignore_wd:
                    i_txes = []
                else:
                    i_txes = a.input_txes.where(lambda t: t.block.height <= height).to_list()
            else:
                o_txes = a.output_txes.to_list()
                if ignore_wd:
                    i_txes = []
                else:
                    i_txes = a.input_txes.to_list()
            clusters[cid]['saddrs'].add(saddr)
            clusters[cid]['addrs'].add(a)
            clusters[cid]['output_txes'].update(o_txes)
            clusters[cid]['input_txes'].update(i_txes)
    return clusters

def art_cluster_with_address(clusters, addr):
    '''
    Return the cluster id containing the given address, or None otherwise. Use
    this only with artificial clusters built using build_art_clusters().
    '''
    for cid, d in clusters.items():
        if addr in d['addrs']:
            return cid
    return None

def multiinput_only(chain, cdir, height=0):
    ''' Generate multi-input only heuristic, with limited height'''
    none_ch = blocksci.heuristics.change.none
    if height:
        cm = blocksci.cluster.ClusterManager.create_clustering(location=cdir,
                chain=chain, stop=height+1, heuristic=none_ch)
    else:
        cm = blocksci.cluster.ClusterManager.create_clustering(location=cdir,
                chain=chain, heuristic=none_ch)
    return cm

def multiinput_change(chain, cdir, height=0):
    '''
    Generate multi-input + client_change_address_behavior heuristics According
    to the docs:
        Most clients will generate a fresh address for the change. If an output
        is the first to send value to an address, it is potentially the change.
    This heuristic will use the unique_change property to cluster outputs only
    when there is one candidate change address output.
    '''
    ab_ch = blocksci.heuristics.change.client_change_address_behavior.unique_change
    stop = (height+1) if height else -1
    cm = blocksci.cluster.ClusterManager.create_clustering(location=cdir,
                chain=chain, stop=stop, heuristic=ab_ch)
    return cm

def multiinput_change_legacy(chain, cdir, height=0):
    ''' Generate multi-input + legacy change address detection heuristics'''
    l_ch = blocksci.heuristics.change.legacy.unique_change
    stop = (height+1) if height else -1
    cm = blocksci.cluster.ClusterManager.create_clustering(location=cdir,
                chain=chain, stop=stop, heuristic=l_ch)
    return cm

def multiinput_reusechange(chain, cdir):
    ''' Generate multi-input + reuse change address detection heuristics'''
    rch = blocksci.heuristics.change.address_reuse()
    cm = blocksci.cluster.ClusterManager.create_clustering(cdir, chain, rch)
    return cm

def load_clusters(chain, cdir):
    try:
        return blocksci.cluster.ClusterManager(cdir, chain)
    except RuntimeError:
        logging.error(f"Cluster folder not found: {cdir}")
        sys.exit(1)

def load_seeds(file, delimiter='\t'):
    seeds = {}
    with open(file, 'r') as f:
        csv_f = csv.DictReader(f, delimiter=delimiter)
        for e in csv_f:
            d = {e['address']: {x: y for x, y in e.items()}}
            seeds.update(d)
    return seeds

def build_load_blocksci(config, height, change=False):
    global cm
    global chain

    try:
        chain = blocksci.Blockchain(config)
    except RuntimeError as e:
        logging.error(e)
        sys.exit(1)

    data_dir = os.path.dirname(config)
    h = f"_{height + 1}" if height else ''
    cdir = f"multi-input_change_legacy{h}" if change else f"multi-input{h}"
    mi = os.path.join(data_dir, "clusters", cdir)

    if os.path.isdir(mi):
        logging.warning(f"Loading clusters from {mi}.")
        cm = load_clusters(chain, mi)
    else:
        logging.warning(f"Producing clusters in {mi}.")
        os.makedirs(mi)
        clustering_heuristic = multiinput_change_legacy if change else multiinput_only
        cm = clustering_heuristic(chain, mi, height)

    return chain, cm

def produce_vectors(saddrs, txes, chain, cm, height=0, label='unknown'):
    """ Produce a list of vectors from either a dict of string addresses
    or a list of blocksci addresses.
    :param saddrs: dict of {Blocksci.address: str_repr} elements
    :param txes: set of all txes of saddrs
    :param cm: ClusterManager instance
    :param height: Maximum height of txes
    :param label: class of these elements
    """
    timer_start = time.time()
    clusters = {}
    vectors = {}
    feature_vectors = []
    timer_txes = time.time()
    t_o = []
    t_i = []
    t_a = []
    for tx in txes:
        # Data per TX
        tx_block_timestamp = tx.block.timestamp
        tx_outs_size = tx.outputs.size
        tx_ins_size = tx.inputs.size
        tx_total_size = tx.total_size
        tx_weight = tx.weight
        tx_fee = tx.fee
        # Convert timestamps to UTC when working with dates
        block_ts = datetime.utcfromtimestamp(tx_block_timestamp)
        is_coinjoin = blocksci.heuristics.is_coinjoin(tx)
        txaddrs = set()

        timer_outs = time.time()
        # Data per address in each input/output slot
        for output in tx.outputs:
            if output.address in saddrs:
                saddr = saddrs[output.address]
                txaddrs.add(saddr)
                v = vectors[saddr] if saddr in vectors else Vector(saddr)
                v.is_out_tx = True
                v.outputs += 1
                v.deposited += output.value
                if height:
                    spent = output.spending_tx_index != None and \
                            output.spending_tx.block_height <= height
                else:
                    spent = output.is_spent
                v.utxos += 1 if not spent else 0
                vectors.update({saddr: v})
        timer_ins = time.time()
        for input in tx.inputs:
            if input.address in saddrs:
                saddr = saddrs[input.address]
                txaddrs.add(saddr)
                v = vectors[saddr] if saddr in vectors else Vector(saddr)
                v.is_in_tx = True
                v.inputs += 1
                v.withdrawn += input.value
                # The inputs at this point in the chain points to earlier txes
                v.inputs_age += input.age
                vectors.update({saddr: v})

        timer_addrs = time.time()
        # Data for all the addresses within this TX
        for txaddr in txaddrs:
            v = vectors[txaddr]
            v.txes_outputs += tx_outs_size
            v.txes_inputs += tx_ins_size
            v.tx_sizes.append(tx_total_size)
            v.tx_weights.append(tx_weight)
            v.tx_fees.append(tx_fee)
            v.activity_days.add(str(block_ts)[:10])
            v.txes_count += 1
            if is_coinjoin:
                v.txes_coinjoin += 1
            if v.is_out_tx:
                v.out_txes_outputs += tx_outs_size
                v.out_txes_inputs += tx_ins_size
                if not v.out_txes_count:
                    v.ts_first_out = tx_block_timestamp
                    v.ts_last_out = tx_block_timestamp
                v.ts_first_out = min(tx_block_timestamp, v.ts_first_out)
                v.ts_last_out = max(tx_block_timestamp, v.ts_last_out)
                v.out_txes_count += 1
                v.activity_days_out.add(str(block_ts)[:10])
                v.txes_years_out[str(block_ts)[:4]] += 1
                if tx.is_coinbase:
                    v.txes_coinbase += 1
                if is_coinjoin:
                    v.txes_coinjoin_out += 1
            if v.is_in_tx:
                v.in_txes_outputs += tx_outs_size
                v.in_txes_inputs += tx_ins_size
                if not v.in_txes_count:
                    v.ts_first_in = tx_block_timestamp
                    v.ts_last_in = tx_block_timestamp
                v.ts_first_in = min(tx_block_timestamp, v.ts_first_in)
                v.ts_last_in = max(tx_block_timestamp, v.ts_last_in)
                v.in_txes_count += 1
                v.activity_days_in.add(str(block_ts)[:10])
                v.txes_years_in[str(block_ts)[:4]] += 1
                if is_coinjoin:
                    v.txes_coinjoin_in += 1
            if v.is_in_tx and v.is_out_tx:
                v.num_txes_same_as_change += 1
            v.is_in_tx = False
            v.is_out_tx = False

        timer_endtx = time.time()
        t_o.append(timer_ins-timer_outs)
        t_i.append(timer_addrs-timer_ins)
        t_a.append(timer_endtx-timer_addrs)

    timer_end_txes = time.time()

    # Resume of all data collected per address
    for addr, saddr in saddrs.items():
        v = vectors[saddr]
        equiv = addr.equiv()
        cluster = cm.cluster_with_address(addr)
        if cluster.index in clusters:
            cluster_addrs = clusters[cluster.index]
        else:
            cluster_addrs = cluster.addresses.size
            clusters.update({cluster.index: cluster_addrs})
        v.cluster = cluster.index
        v.clust_addrs = cluster_addrs
        v.eq_addrs = equiv.addresses.size
        vector = {
            'label': label,
            'address': saddr,
            'cluster_id': v.cluster,
            'datetime_first': v.datetime_first(),
            'datetime_last': v.datetime_last(),
            'timestamp_first': v.ts_first(),
            'timestamp_last': v.ts_last(),
            'timestamp_first_out': v.ts_first_out,
            'timestamp_last_out': v.ts_last_out,
            'timestamp_first_in': v.ts_first_in if v.ts_first_in else 0,
            'timestamp_last_in': v.ts_last_in if v.ts_last_in else 0,
            'lifetime': v.lifetime(),
            'timespan_d': v.timespan_out(),
            'timespan_w': v.timespan_in(),
            'type': addr.full_type,
            'balance': v.balance(),
            'deposited': v.deposited,
            'withdrawn': v.withdrawn,
            'utxos': v.utxos,
            'outputs': v.outputs,
            'inputs': v.inputs,
            'txes': v.txes_count,
            'txes_out': v.out_txes_count,
            'txes_in': v.in_txes_count,
            'tx_size_mean': sum(v.tx_sizes) / len(v.tx_sizes),
            'tx_weight_mean': sum(v.tx_weights) / len(v.tx_weights),
            'tx_fee_mean': sum(v.tx_fees) / len(v.tx_fees),
            'outs_per_tx': v.txes_outputs / v.txes_count,
            'ins_per_tx': v.txes_inputs / v.txes_count,
            'outs_per_out_tx': v.out_txes_outputs / v.out_txes_count,
            'ins_per_out_tx': v.out_txes_inputs / v.out_txes_count,
            'outs_per_in_tx': v.in_txes_outputs / v.in_txes_count \
                    if v.in_txes_count else -1,
            'ins_per_in_tx': v.in_txes_inputs / v.in_txes_count \
                    if v.in_txes_count else -1,
            'ins_age_mean': v.inputs_age / v.inputs if v.inputs else -1,
            'coinbase': v.txes_coinbase,
            'coinjoin': v.txes_coinjoin,
            'coinjoin_out': v.txes_coinjoin_out,
            'coinjoin_in': v.txes_coinjoin_in,
            'cluster_size': v.clust_addrs,
            'equiv_addrs': v.eq_addrs,
            'activity': v.activity(),
            'activity_d': v.activity_d(),
            'activity_w': v.activity_w(),
            'idle_days': v.idle_days(),
            'profit_rate': v.deposited / v.lifetime() if v.lifetime() else -1,
            'expense_rate': v.withdrawn / v.lifetime() if v.lifetime() else -1,
            'daily_d_rate': v.daily_d_rate(),
            'daily_w_rate': v.daily_w_rate(),
            'd_per_tx': v.d_per_tx(),
            'w_per_tx': v.w_per_tx(),
            'tx_ratio': v.tx_ratio(),
            'yearly_d_txes': v.yearly_d_txes(),
            'yearly_w_txes': v.yearly_w_txes(),
            'addr_as_change': v.addr_as_change(),
            }
        feature_vectors.append(vector)

    timer_end_resume = time.time()
    timer_end = time.time() - timer_start
    logging.info(f"search txes:\t{timer_txes-timer_start}")
    logging.info(f"txes:\t{timer_end_txes-timer_txes}")
    logging.info(f"[outs:\t{sum(t_o)}\tins: {sum(t_i)}\taddrs: {sum(t_a)}]")
    logging.info(f"resume:\t{timer_end_resume-timer_end_txes}")
    logging.info(f"all:\t{timer_end}")
    return feature_vectors, timer_end

def produce_vector(addr, chain, cm, txes=[], height=0, label='unknown'):
    timer_start = time.time()
    if isinstance(addr, str):
        saddr = addr
        addr, txes = addr_txes_from_string(saddr, chain, height)
    else:
        saddr = addr_to_string(addr)
    if not addr or not txes:
        return None, time.time()-timer_start
#    saddr = addr_to_string(addr)
    txes_count = 0
    out_txes_count = 0
    in_txes_count = 0
    txes_outputs = 0
    in_txes_outputs = 0
    out_txes_outputs = 0
    txes_inputs = 0
    in_txes_inputs = 0
    out_txes_inputs = 0
    outputs = 0
    inputs = 0
    utxos = 0
    inputs_age = 0
    tx_sizes = []
    tx_weights = []
    tx_fees = []
    txes_coinbase = 0
    txes_coinjoin = 0
    txes_coinjoin_out = 0
    txes_coinjoin_in = 0
    withdrawn = 0
    deposited = 0
    ts_first_in = None
    ts_last_in = None
    activity_days = set()
    activity_days_out = set()
    activity_days_in = set()
    txes_years_out = defaultdict(int)
    txes_years_in = defaultdict(int)
    num_txes_same_as_change = 0
    for tx in txes:
        is_out_tx = False
        is_in_tx = False
        tx_block_timestamp = tx.block.timestamp
        tx_outs_size = tx.outputs.size
        tx_ins_size = tx.inputs.size
        tx_total_size = tx.total_size
        tx_weight = tx.weight
        tx_fee = tx.fee
        txes_outputs += tx_outs_size
        txes_inputs += tx_ins_size
        for output in tx.outputs.to_list():
            if addr == output.address:
                is_out_tx = True
                outputs += 1
                deposited += output.value
                if height:
                    spent = output.spending_tx_index != None and \
                            output.spending_tx.block_height <= height
                else:
                    spent = output.is_spent
                utxos += 1 if not spent else 0
        for input in tx.inputs.to_list():
            if addr == input.address:
                is_in_tx = True
                inputs += 1
                withdrawn += input.value
                # The inputs at this point in the chain points to earlier txes
                inputs_age += input.age
        is_coinjoin = blocksci.heuristics.is_coinjoin(tx)
        if is_coinjoin:
            txes_coinjoin += 1
        # Convert timestamps to UTC when working with dates
        block_ts = datetime.utcfromtimestamp(tx.block.timestamp)
        activity_days.add(str(block_ts)[:10])
        tx_sizes.append(tx_total_size)
        tx_weights.append(tx_weight)
        tx_fees.append(tx_fee)
        txes_count += 1
        if is_in_tx and is_out_tx:
            num_txes_same_as_change += 1

        if is_out_tx:
            out_txes_outputs += tx_outs_size
            out_txes_inputs += tx_ins_size
            if not out_txes_count:
                ts_first_out = tx_block_timestamp
                ts_last_out = tx_block_timestamp
            ts_first_out = min(tx_block_timestamp, ts_first_out)
            ts_last_out = max(tx_block_timestamp, ts_last_out)
            out_txes_count += 1
            if tx.is_coinbase:
                txes_coinbase += 1
            if is_coinjoin:
                txes_coinjoin_out += 1
            activity_days_out.add(str(block_ts)[:10])
            txes_years_out[str(block_ts)[:4]] += 1

        if is_in_tx:
            in_txes_outputs += tx_outs_size
            in_txes_inputs += tx_ins_size
            if not in_txes_count:
                ts_first_in = tx_block_timestamp
                ts_last_in = tx_block_timestamp
            ts_first_in = min(tx_block_timestamp, ts_first_in)
            ts_last_in = max(tx_block_timestamp, ts_last_in)
            in_txes_count += 1
            if is_coinjoin:
                txes_coinjoin_in += 1
            activity_days_in.add(str(block_ts)[:10])
            txes_years_in[str(block_ts)[:4]] += 1

    activity = len(activity_days)
    activity_d = len(activity_days_out)
    activity_w = len(activity_days_in)
    tx_ratio = in_txes_count/out_txes_count
    d_per_tx = deposited/out_txes_count
    w_per_tx = withdrawn/in_txes_count if in_txes_count else -1
    addr_as_change = num_txes_same_as_change / txes_count
    ts_first = min(ts_first_out, ts_last_out)
    all_ts = [ts_first_out, ts_last_out, ts_first_in, ts_last_in]
    ts_last = max([ts for ts in all_ts if ts])
    lifetime = ts_last - ts_first
    total_days = int((ts_last - ts_first) / (3600*24))
    daily_d_rate = out_txes_count/total_days if total_days else -1
    daily_w_rate = in_txes_count/total_days if total_days else -1
    idle_days = total_days - activity
    timespan_out = ts_last_out - ts_first_out
    timespan_in = ts_last_in - ts_first_in if ts_first_in else 0
    yearly_d_txes = sum(txes_years_out.values())/len(txes_years_out)
    yearly_w_txes = sum(txes_years_in.values())/len(txes_years_in) \
            if txes_years_in else -1
    cluster = cm.cluster_with_address(addr)
    clust_addrs = cluster.addresses.to_list()
    equiv = addr.equiv()
    eq_addrs_list = equiv.addresses.to_list()
    vector = {
            'label': label,
            'address': saddr,
            'cluster_id': cluster.index,
            'datetime_first': datetime.utcfromtimestamp(ts_first),
            'datetime_last': datetime.utcfromtimestamp(ts_last),
            'timestamp_first': ts_first,
            'timestamp_last': ts_last,
            'timestamp_first_out': ts_first_out,
            'timestamp_last_out': ts_last_out,
            'timestamp_first_in': ts_first_in if ts_first_in else 0,
            'timestamp_last_in': ts_last_in if ts_last_in else 0,
            'lifetime': lifetime,
            'timespan_d': timespan_out,
            'timespan_w': timespan_in,
            'type': addr.full_type,
            'balance': addr.balance(height) if height else addr.balance(),
            'deposited': deposited,
            'withdrawn': withdrawn,
            'utxos': utxos,
            'outputs': outputs,
            'inputs': inputs,
            'txes': txes_count,
            'txes_out': out_txes_count,
            'txes_in': in_txes_count,
            'tx_size_mean': sum(tx_sizes) / len(tx_sizes),
            'tx_weight_mean': sum(tx_weights) / len(tx_weights),
            'tx_fee_mean': sum(tx_fees) / len(tx_fees),
            'outs_per_tx': txes_outputs / txes_count,
            'ins_per_tx': txes_inputs / txes_count,
            'outs_per_out_tx': out_txes_outputs / out_txes_count,
            'ins_per_out_tx': out_txes_inputs / out_txes_count,
            'outs_per_in_tx': in_txes_outputs / in_txes_count \
                    if in_txes_count else -1,
            'ins_per_in_tx': in_txes_inputs / in_txes_count \
                    if in_txes_count else -1,
            'ins_age_mean': inputs_age / inputs if inputs else -1,
            'coinbase': txes_coinbase,
            'coinjoin': txes_coinjoin,
            'coinjoin_out': txes_coinjoin_out,
            'coinjoin_in': txes_coinjoin_in,
            'cluster_size': len(clust_addrs),
            'equiv_addrs': len(eq_addrs_list),
            'activity': activity,
            'activity_d': activity_d,
            'activity_w': activity_w,
            'idle_days': idle_days,
            'profit_rate': deposited / lifetime if lifetime else -1,
            'expense_rate': withdrawn / lifetime if lifetime else -1,
            'daily_d_rate': daily_d_rate,
            'daily_w_rate': daily_w_rate,
            'd_per_tx':  d_per_tx,
            'w_per_tx': w_per_tx,
            'tx_ratio': tx_ratio,
            'yearly_d_txes': yearly_d_txes,
            'yearly_w_txes': yearly_w_txes,
            'addr_as_change': addr_as_change,
            }
    timer_end = time.time() - timer_start
    return vector, timer_end

def is_force_addr_reuse_tx(tx, max_u_am=4, min_outs=100):
    '''
    Check if the different BTC amounts sent are just a few compared to the
    number of outputs. This is a characteristic behavior of force address reuse
    attack and some forms of "advertising".
    :param max_u_am: maximum number of different amounts of BTC sent
    :param min_outs: minimum number of destination addresses with similar
        BTC amounts
    '''
    # TODO This heuristic can be improved by checking if the value of the
    # outputs is lower/close than the dust threshold, actually 546 satoshis for
    # non-segwit and 294 for segwit txes, at the default rate of 3000 sat/kB
#    return (len(set(tx.outputs.value)) < max_u_am and \
#            tx.outputs.size > min_outs) \
#            or tx.outputs.size > max_outs
    return len(set(tx.outputs.value)) < max_u_am and tx.outputs.size > min_outs

def force_addr_reuse_txes(txes):
    far = [tx for tx in txes if is_force_addr_reuse_tx(tx)]
    return far

# Could also use blocksci.heuristics.is_definite_coinjoin
# or blocksci.heuristics.possible_coinjoin_txes
def coinjoin_txes(txes):
    cjs = [tx for tx in txes if blocksci.heuristics.is_coinjoin(tx)]
    return cjs

def coinbase_txes(txes):
    coinbase = [tx for tx in txes if tx.is_coinbase]
    return coinbase

def ts_first_tx(txes):
    return min([tx.block_time for tx in txes], default=None)

def ts_last_tx(txes):
    return max([tx.block_time for tx in txes], default=None)

def total_slots_values(slots):
    return sum([s.value for s in slots])

def utxos(outs):
    return sum([1 for o in outs if not o.is_spent])

def activity_days(txes):
    # Format datetime as 'YYYY-mm-dd'
#    d = [dt.strftime('%Y-%m-%d') for dt in txes.block_time]
    return len(set([str(dt)[:10] for dt in txes.block_time]))

def yearly_txes_mean(txes):
    counter = defaultdict(int)
    for dt in [str(dt)[:4] for dt in txes.block_time]:
        counter[dt] += 1
    elements = [v for k, v in counter.items()]
    return sum(elements) / len(elements) if elements else -1

# tx total size: tx size in bytes serialized, including base/witness data (as
# defined in BIP144)
def tx_size_mean(txes, n):
    return sum([tx.total_size for tx in txes]) / n

# tx weight: three times the base size plus the total size
def tx_weight_mean(txes, n):
    return sum([tx.weight for tx in txes]) / n

def tx_fee_mean(txes, n):
    return sum([tx.fee for tx in txes]) / n

def outputs_per_tx(txes):
    # TODO after the list comprehension, txes.size becomes 0, seems buggy
    txes_outs = [tx.outputs.size for tx in txes]
    return sum(txes_outs) / len(txes_outs) if txes_outs else -1

def inputs_per_tx(txes):
    txes_ins = [tx.inputs.size for tx in txes]
    return sum(txes_ins) / len(txes_ins) if txes_ins else -1

def inputs_age_mean(ages):
    return sum(ages) / len(ages) if ages.any() else -1

def same_address_as_exchange(addr):
    changes = [tx for tx in addr.in_txes if addr in tx.outputs.address.to_list()]
    return len(changes) / addr.txes.size

def address_in_out_txes(addr, height=0, filter_coinjoin=True, filter_far=True):
    # TODO is it necessary to check txes of equiv addreses here?
    td = addr.output_txes.where(lambda t: t.block.height <= height).to_list()\
        if height else addr.output_txes.to_list()
    td = set(td)
    tw = addr.input_txes.where(lambda t: t.block.height <= height).to_list()\
        if height else addr.input_txes.to_list()
    tw = set(tw)
    coinjoin = set()
    if filter_coinjoin:
        cjd = set(coinjoin_txes(td))
        if cjd:
            logging.info(f"CoinJoin deposits in {addr}: {cjd}")
            td -= cjd
        cjw = set(coinjoin_txes(tw))
        if cjw:
            logging.info(f"CoinJoin withdrawals in {addr}: {cjw}")
            tw -= cjw
        coinjoin = cjd | cjw
    far = set()
    if filter_far:
        ard = set(force_addr_reuse_txes(td))
        if ard:
            logging.info(f"Force-address-reuse deposits in {addr}: {ard}")
            td -= ard
        arw = set(force_addr_reuse_txes(tw))
        if arw:
            logging.info(f"Force-address-reuse withdrawals in {addr}: {arw}")
            tw -= arw
        far = ard | arw
    txes = td | tw
    exp_rank = sum([(t.input_count + t.output_count) for t in txes])
    return {'d_txes': td, 'w_txes': tw, 'txes': txes,
            'coinjoin': coinjoin, 'far': far, 'exp_rank': exp_rank,
            'total': len(txes) + len(coinjoin) + len(far)}

def addresses_txes(addrs, height=0, filter_coinjoin=True, filter_far=True):
    addr_txes = {a: address_in_out_txes(a, height, filter_coinjoin, filter_far)\
            for a in addrs}
    return addr_txes

def blacklisted_from_string(addr, chain):
    try:
        bs_addr = chain.address_from_string(addr)
    except Exception as e:
        logging.debug(f"Error triggered by {addr}: {repr(e)}")
        return None
    if not bs_addr:
        logging.debug(f"Address not found in the blockchain: {addr}")
        return None
    return bs_addr

def addr_in_out_txes_from_string(addr, chain, height=0, filter_coinjoin=True,
        filter_far=True):
    try:
        bs_addr = chain.address_from_string(addr)
    except Exception as e:
        logging.debug(f"[!] Error triggered by {addr}: {repr(e)}")
        return None, None
    if not bs_addr:
        logging.debug(f"Address not found in the blockchain: {addr}")
        return None, None
    # Sometimes we have an addr that doesn't have txes, although it is
    # retrieved by blocksci 'cause its pubkey appears in the blockchain with a
    # different encoding (i.e. some equiv addr). This is also the case for some
    # pay-to-pubkeyhash encoded addrs that have txes attached only as 
    # pay-to-pubkey (e.g. 1DGVoJo5Pkoju3VpvNkhRHQnHWksWUR2Sp). This seems to be
    # done on purpose (p2pk and p2pkh have the same encoding in blocksci), see
    # https://github.com/citp/BlockSci/issues/365#issuecomment-578545533
    if not bs_addr.txes.size:
        eaddrs = bs_addr.equiv().addresses.to_list()
        if len(eaddrs) > 1:
            logging.warning(f"{addr} has more than one equiv address.")
        # TODO should we return all equiv addrs instead of just one?
        for ea in eaddrs:
            # TODO blocksci.MultisigPubkey:112GNsEeSqroP5ByyXN6skYzvzUsCz5TcY
            # produces a segmentation fault when accessing ea.txes object, so
            # we use this another method to check for a valid addr (output txes
            # count should suffice, checking input_txes just in case.
            if ea.output_txes_count() or ea.input_txes_count():
                bs_addr = ea
    if height:
        td = set(bs_addr.output_txes.where(lambda t:
            t.block.height <= height).to_list())
        tw = set(bs_addr.input_txes.where(lambda t:
            t.block.height <= height).to_list())
        m = f" at height {height}"
    else:
        td = set(bs_addr.output_txes.to_list())
        tw = set(bs_addr.input_txes.to_list())
        m = ''
    if not td and not tw:
        logging.debug(f"{addr} does not have any TX{m}. Skipping.")
        return None, None
    # Filter CoinJoin and Force-Address-Reuse txes
    coinjoin = set()
    far = set()
    if filter_coinjoin:
        cjd = set(coinjoin_txes(td))
        cjw = set(coinjoin_txes(tw))
        if cjd:
            logging.info(f"CoinJoin deposits in {addr}: {cjd}")
            td -= cjd
        if cjw:
            logging.info(f"CoinJoin withdrawals in {addr}: {cjw}")
            tw -= cjw
        coinjoin = cjd | cjw
    if filter_far:
        ard = set(force_addr_reuse_txes(td))
        arw = set(force_addr_reuse_txes(tw))
        if ard:
            logging.info(f"Force-address-reuse deposits in {addr}: {ard}")
            td -= ard
        if arw:
            logging.info(f"Force-address-reuse withdrawals in {addr}: {arw}")
            tw -= arw
        far = ard | arw
    txes = td | tw
    # Explosion Rank to calculate how many nodes could introduce this address
    # into the graph, it's approximate due to addresses count twice if repeated
    exp_rank = sum([(t.input_count + t.output_count) for t in txes])
    return bs_addr, {'d_txes': td, 'w_txes': tw, 'txes': txes,
            'coinjoin': coinjoin, 'far': far, 'exp_rank': exp_rank,
            'total': len(txes) + len(coinjoin) + len(far)}

def addr_txes_from_string(addr, chain, height=0):
    try:
        bs_addr = chain.address_from_string(addr)
    except Exception as e:
        logging.debug(f"Error triggered by {addr}: {repr(e)}")
        return None, None
    if not bs_addr:
        logging.debug(f"Address not found in the blockchain: {addr}")
        return None, None
    # Fix for empty equiv address (i.e. pubkeyhash instead pubkey)
    if not bs_addr.txes.size:
        eaddrs = bs_addr.equiv().addresses.to_list()
        if len(eaddrs) > 1:
            logging.warning(f"{addr} has more than one equiv address.")
        for ea in eaddrs:
            if ea and ea.txes.size:
                bs_addr = ea
    if height:
        txes = bs_addr.txes.where(lambda t: t.block.height <= height).to_list()
        if not txes:
            logging.debug(f"{addr} does not have any TX at height {height}. Skipping.")
            return None, None
    else:
        txes = bs_addr.txes.to_list()
        if not txes:
            logging.debug(f"{addr} does not have any TX. Skipping.")
            return None, None
    return bs_addr, txes

def addr_from_string(addr, chain, height=0):
    try:
        bs_addr = chain.address_from_string(addr)
    except Exception as e:
        logging.debug(f"Error triggered by {addr}: {repr(e)}")
        return None
    if not bs_addr:
        logging.debug(f"Address not found in the blockchain: {addr}")
        return None
    # Fix for empty equiv address (i.e. pubkeyhash instead pubkey)
    if not bs_addr.txes.size:
        eaddrs = bs_addr.equiv().addresses.to_list()
        if len(eaddrs) > 1:
            logging.warning(f"{addr} has more than one equiv address.")
        for ea in eaddrs:
            if ea and ea.txes.size:
                bs_addr = ea
    if height:
        if not bs_addr.txes.where(lambda t: t.block.height <= height).size:
            logging.debug(f"{addr} does not have any TX at height {height}. Skipping.")
            return None
    else:
        if not bs_addr.txes.size:
            logging.debug(f"{addr} does not have any TX. Skipping.")
            return None
    return bs_addr

def addr_to_string(addr):
    # Raw Multisig doesn't have address format
    if type(addr) == blocksci.MultisigAddress:
        return f"{[a for a in addr.addresses]}"
    # Return input|output script for non standards
    elif type(addr) == blocksci.NonStandardAddress:
        return f"{addr.in_script}|{addr.out_script}"
    # WitnessUnknownAddress has no attribute address_string
    elif type(addr) == blocksci.WitnessUnknownAddress:
        return f"{addr.address_num}:{addr.witness_script}"
    # Return data for OP_RETURN scripts
    elif type(addr) == blocksci.OpReturn:
        return addr.data.hex()
    else:
        return addr.address_string

def extract_features_range(addrs, chain, cm, height=0, label='unknown'):
    msj = f"Extracting features of {len(addrs)} addresses\n"
    saddrs = {}
    txes = set()
    test = next(iter(addrs))
    if isinstance(test, str):
        # Addresses are strings
        for saddr, d in addrs.items():
            addr, atxes = addr_txes_from_string(saddr, chain, height)
            if not addr:
                continue
            saddrs.update({addr: saddr})
            txes.update(atxes)
    elif isinstance(test, blocksci.Address):
        # Addresses are BlockSci.address instances
        for addr, atxes in addrs.items():
            if not atxes:
                if height:
                    atxes = addr.txes.where(lambda t:
                            t.block.height <= height).to_list()
                else:
                    atxes = addr.txes.to_list()
            saddrs.update({addr: addr_to_string(addr)})
            txes.update(atxes)
    else:
        msj += "The list should contain either str or blocksci.Address "
        msj += "objects. Aborting."
        logging.debug(msj)
        return []
    v, t = produce_vectors(saddrs, txes, chain, cm, height, label)
    msj += f"Time elapsed extracting features: {t}\n"
    logging.info(msj)
    return v

def extract_features(seed, chain, cm, height=0, label='unknown'):
    '''Return all addresses in the cluster where seed is found'''
    # The cluster does not includes all equiv addresses clusters.
    # i.e. if a pubkey is used to co-spent with another pubkey, and it is also
    # part of a multisig address, two clusters are created whilest such two
    # addresses are not co-spent together.
    msj = f"Extracting features for {seed}\n"
    v, t = produce_vector(seed, chain, cm, txes=[], height=height, label=label)
    msj += f"Time elapsed extracting features for address {seed}: {t}\n"
    logging.info(msj)
    return v

def predict(estimator, dataset, height, bycluster=True, tag_filter=True, debug=False):
    X = dataset.drop(['label'], axis=1)
    prediction = {}
    if bycluster:
        cids = dataset['cluster_id'].unique()
        for cid in cids:
            if tag_filter:
                owner, ctag, csize = tags.search_tag_by_cluster(cid=cid, height=height)
                if tags.is_service_owner(owner):
                    msg = f"CID:{cid}: is a service {owner}>>{ctag}. Skipping."
                    logging.info(msg)
                    prediction[cid] = -1
                    continue
            mask = X['cluster_id'] == cid
            idx = X[mask].index
            cresult = estimator.predict_proba(X[mask])

            # Average probability
            avg = sum([r[0] for r in cresult]) / len(cresult)
            cclass = 'Exchange' if avg > 0.5 else 'Non-exchange'
            logging.info(f"AVGPROB CID:{cid}: {avg} => {cclass}")
            prediction[cid] = avg

            # Majority voting
            avg = sum([1 for r in cresult if r[0] > 0.5]) / len(cresult)
            cclass = 'Exchange' if avg > 0.5 else 'Non-exchange'
            logging.info(f"MAJVOTE CID:{cid}: {avg} => {cclass}")

            # One positive voting
            avg = sum([1 for r in cresult if r[0] > 0.5])
            cclass = 'Exchange' if avg > 0 else 'Non-exchange'
            logging.info(f"INDVOTE CID:{cid}: {avg} => {cclass}")

            # Individual results
            if debug:
                prediction_results(cresult, dataset, idx)
    else:
        results = estimator.predict_proba(X)
        prediction['all'] = results
        if debug:
            # TODO check the idx value
            prediction_results(results, dataset, dataset.index)
    return prediction

def prediction_results(results, dataset, idx):
    for i, n in enumerate(idx):
        iclass = 'Exchange' if results[i][0] > 0.5 else 'Non-exchange'
        addr = dataset.iloc[n]['address']
        cid = dataset.iloc[n]['cluster_id']
        logging.info(f"{cid}:{addr}: {results[i]} => {iclass}")

def address_feature_extraction(addr, height):
    global cm
    global chain

    vectors = extract_features_range({addr: None}, chain, cm, height)

    return pd.DataFrame(data=vectors, columns=vectors[0].keys())

def clusters_feature_extraction(cids, height, skip_services=True):
    global cm
    global chain

    g_vectors = []
    for cid in cids:
        c = cm.clusters()[cid]
        owner, ctag, csize = tags.search_tag_by_cluster(cid=cid, height=height)
        if skip_services and tags.is_service_owner(owner):
            m = f"CID:{cid}: is a service {owner}>>{ctag}. Skipping."
            logging.info(m)
            continue

        addrs = {a: None for a in c.addresses.to_list()}
        vectors = extract_features_range(addrs, chain, cm, height)
        g_vectors.extend(vectors)

    if g_vectors:
        return pd.DataFrame(data=g_vectors, columns=g_vectors[0].keys())
    else:
        return None

def bitmex_classifier(addr, chain):
    global bmex_addrs
    if not bmex_addrs:
        # We use as reference the cold-wallet of BitMEX, obtained from 
        # https://bitinfocharts.com/bitcoin/wallet/BITMEX-coldwallet
        bmex = chain.address_from_string('3BMEXqGpG4FxBA1KWhRFufXfSTRgzfDBhJ')
        bmex_addrs = set(bmex.wrapped_address.addresses.to_list())
    if addr.full_type == 'scripthash/multisig3of4':
        addrs = set(addr.wrapped_address.addresses.to_list())
        return len(addrs.intersection(bmex_addrs)) >= 3
    else:
        return False

def expand(seed, chain, cm, clusters, height, ticker):
    '''Return all addresses in the cluster where seed is found'''
    import tags
    logging.info(f"Expanding {addr_to_string(seed)} of type {seed.type}")
    cluster = cm.cluster_with_address(seed)
    cid = cluster.index
    # TODO Avoid expanding service-tagged clusters
    owner, ctag, csize = tags.search_tag_by_cluster(cid=cid, chain=chain,
            cm=cm, height=height, ticker=ticker)
    logging.info(f"\tCTag: {ctag}")
#    if tags.is_service_ctag(ctag):
#        logging.info(f"\tCluster {cid} is a service. Skipping.")
#        return cid, [], clusters
    if cid in clusters:
        logging.info(f"\tCluster {cid} is already included")
        return cid, [], clusters
    clusters.add(cid)
    caddrs = cluster.addresses.to_list()
    logging.info(f"\tRaw address: {seed}")
    logging.info(f"\tCluster: {cid}")
    logging.info(f"\tSize {len(caddrs)}")
    return cid, caddrs, clusters


def main(args):
    global chain, cm

    # Load the blockchain
    chain, cm = build_load_blocksci(args.blocksci, args.height)

    seeds = load_seeds(args.inputfile)

    # Expand the initial seeds
    if args.expand:
        exp_seeds = {}
        clusters = set()
        for seed in seeds:
            addr, atxes = addr_txes_from_string(seed, chain, args.height)
            if not addr:
                continue
            cid, caddrs, clusters = expand(addr, chain, cm, clusters,
                    args.height, args.ticker)
            if caddrs:
                exp_seeds.update({a: None for a in caddrs})
            exp_seeds.update({addr: atxes})
        if not exp_seeds:
            logging.info(f"No data available.")
            exit(-1)
        seeds = exp_seeds

    # Extract feature vectors from a set of addresses
    vectors = extract_features_range(seeds, chain, cm, args.height, args.label)
    with open(args.output, 'w', 1) as fo:
        for i, vector in enumerate(vectors):
            if not i:
                # First write headers
                fo.write("\t".join(vector.keys()) + "\n")
            fo.write("\t".join([f"{v}" for k, v in vector.items()]) + "\n")

if __name__ == '__main__':
    usage = "\n\n\tExtract a set of features from a list of bitcoin" +\
            " addresses, and write the results into a file."
    version = '2.4.2'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('-D', '--blocksci', dest='blocksci', action='store',
            type=str, help='Blocksci config file')
    parser.add_argument('-i', '--inputfile', dest='inputfile', action='store',
            type=str, help='File having addresses for feature extraction')
    parser.add_argument('-o', '--output', dest='output', action='store',
            type=str, default='features.tsv', help='Name for the output file')
    parser.add_argument('-l', '--label', dest='label', action='store',
            type=str, default='unknown', help='Label (class) for this dataset')
    parser.add_argument('-H', '--height', dest='height', action='store',
            type=int, default=0, help='Max block height')
    parser.add_argument('-e', '--expand', dest='expand', action='store_true',
            default=False, help='Use multi-input to expand initial seeds')
    parser.add_argument('-t', '--ticker', dest='ticker', action='store',
            type=str, default='btc', choices=['btc', 'bch', 'ltc'],
            help='Blockchain to use, default=btc')
    parser.add_argument('-v', '--version', action='version', version=version)

    args = parser.parse_args()

    if not args.blocksci or not os.path.isfile(args.blocksci):
        e = "Blocksci config file does not exist."
        parser.error(e)
    if not args.inputfile or not os.path.isfile(args.inputfile):
        e = "Input file does not exist."
        parser.error(e)

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Made with version {version}")
    logging.debug(f"- added support for change-address clustering heuristic")
    logging.debug(f"- added support for art-clusters")
    logging.debug(f"- changed OpReturn string repr (now is data in hex)")
    logging.debug(f"- fixed segfault from buggy equiv addr")
    logging.debug(f"{args}")

    main(args)

