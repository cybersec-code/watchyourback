import logging
import pandas as pd
import networkx as nx
from joblib import load
import copy, blocksci, json, argparse, os
from datetime import datetime
from collections import defaultdict
try:
    from lib import blockchain_feature_extraction as bfe
    from lib.tags import search_tag_by_cluster as search_tag
    from lib import price_api as price
    from lib.Entity import Entity
except ModuleNotFoundError:
    import blockchain_feature_extraction as bfe
    from tags import search_tag_by_cluster as search_tag
    import price_api as price
    from Entity import Entity

chain = None
cm = None

def explore_clusters(height, merge, clusters, art_c, txes, output, operation,
        ignore_wd=True, save=True, classifier=None):
    merged_clusters = {}
    graphs = []
    cids = []
    cowners = []
    ctagss = []
    csizes = 0
    g_cj_sent = 0
    g_cj_recv = 0
    explored = []

    for cid in clusters:
        logging.info(f"Exploring cluster: {cid}")
        entity, e_clusters = \
                explore_cluster(cid, height, merge, clusters, art_c, ignore_wd,
                        classifier)

        if merge:
            merged_clusters = merge_clusters(merged_clusters, e_clusters)
            cids.append(cid)
            cowners.append(entity.owner)
            ctagss.append(entity.ctags)
            csizes += entity.size
            g_cj_sent += entity.coinjoin_txes_out_count
            g_cj_recv += entity.coinjoin_txes_in_count

        if save:
            # Save into JSON file
            e_clusters = to_json(entity, e_clusters, txes, output)

            # Produce a Graph
            G = to_graph(entity, e_clusters)
            ofile = os.path.join(output, f"{cid}.gml")
            nx.readwrite.gml.write_gml(G, ofile)
            graphs.append(G)
        else:
            explored.append(e_clusters)

    if merge:
        if operation:
            cid = operation
        else:
            cid = ",".join(sorted([str(c) for c in cids]))
        cowner = "|".join(sorted(set(cowners)))
        ctags = "|".join(sorted(set(ctagss)))
        entity.cid = cid
        entity.owner = cowner
        entity.ctags = ctags
        entity.size = csizes
        entity.coinjoin_txes_out_count = g_cj_sent
        entity.coinjoin_txes_in_count = g_cj_recv
        if save:
            # dump JSON/graph files
            to_json(entity, merged_clusters, txes, output)
            G = graph_compose_from_list(graphs)
            ofile = os.path.join(output, f"{cid}.gml")
            nx.readwrite.gml.write_gml(G, ofile)
        else:
            return merged_clusters

    return explored

def explore_cluster(cid, height, merge, cseeds, art_c=None, ignore_wd=True,
        classifier=None):
    global chain, cm
    if chain is None:
        if bfe.chain is None and bfe.cm is None:
            logging.error("Instances of Blocksci not found.")
            return
        else:
            logging.warning("Loading instances of Blocksci from bfe.")
            cm = bfe.cm
            chain = bfe.chain

    if art_c:
        owner, ctags = '', ''
    else:
        owner, ctags, _ = search_tag(cid=cid, height=height)

    entity = Entity(cid, owner, ctags, art_c=art_c, cm=cm, height=height,
            ignore_wd=ignore_wd)
    # TODO TESTING: Use the exchange classifier in case this may be an exchange
    entity.classify(classifier, chain, cm)
    e_clusters = defaultdict(dict)
    d_clusters = set()
    w_clusters = set()

    # Backwards

    for t in entity.output_txes:
        if t.is_coinbase:
            outs = {(t, o, 1) for o in t.outputs if entity.has_addr(o.address)}
            d = {
                    'outs': outs,
                    'addrs_sending': [],
                    'addrs_receiving': {
                        f"{cid}:{bfe.addr_to_string(o.address)}"
                        for (t, o, p) in outs
                    },
                }
            # TODO use a value different than -1
            e_clusters = update_outs(-1, d, e_clusters, height, True)
            continue

        # Some very rare txes may have cero inputs, like this one:
        # a842b87403e6d6ca1a9ea39b16d496ebeb6ab15b83acb619cc10daba08114029
        if t.input_value == 0:
            logging.warning(f"Empty tx: {t.hash}\tinputs: {t.inputs.to_list()}\t{t.input_value} BTCs")
            continue

        d_cids = defaultdict(set)
        coinjoin = blocksci.heuristics.is_coinjoin(t)
        # There may be some senders from different clusters when using art_c
        if coinjoin or art_c:

            if coinjoin:
                entity.update_coinjoin_txes_out({str(t.hash)})
                logging.debug(f"COINJOIN DEPOSIT: {t.hash}")

            # Identify clusters sending funds, and the total amount sent
            sender = defaultdict(int)
            sender_addrs = defaultdict(set)
            total_deposited = 0
            for i in t.inputs.to_list():
                d_cid = get_cluster(i.address, art_c, cm)
                d_value = i.value
                sender[d_cid] += d_value
                total_deposited += d_value
                saddr = f"{d_cid}:{bfe.addr_to_string(i.address)}"
                sender_addrs[d_cid].add(saddr)

            try:
                # Replace the amount sent with the contribution percentage
                sender = {k: v/total_deposited for k, v in sender.items()}
            except ZeroDivisionError as ex:
                logging.error(f"ZeroDivisionError in a deposit")
                logging.error(f"tx: {t.hash}\tinputs: {t.inputs.to_list()}\tsender: {sender}\ttotal_deposited: {total_deposited}")
                raise ex

            # Values being deposited to the cluster
            outs = [(t, o) for o in t.outputs if entity.has_addr(o.address)]

            # Adjust the amounts sent to the cluster proportionally with
            # respect to each sending cluster's contribution p
            for d_cid, p in sender.items():

                # Avoid reporting values deposited by the same cluster
                if d_cid == cid:
                    continue

                # Avoid reporting values deposited by the same seed clusters
                elif merge and d_cid in cseeds:
                    continue

                if d_cid not in d_cids:
                    d_cids[d_cid] = {
                            'outs': set(),
                            'addrs_sending': sender_addrs[d_cid],
                            'addrs_receiving': set()
                        }

                d_cids[d_cid]['outs'].update([(t, o, p) for (t, o) in outs])
                d_cids[d_cid]['addrs_receiving'].update([
                        f"{cid}:{bfe.addr_to_string(o.address)}"
                        for (t, o) in outs
                    ])

        else:
            # Identify the cluster sending funds
            inputs = t.inputs.to_list()
            input_addrs = t.inputs.address.to_list()
            d_cid = get_cluster(input_addrs[0], art_c, cm)

            # Avoid reporting values deposited by the same cluster
            if d_cid == cid:
                continue

            # Avoid reporting values deposited by the same seed clusters
            elif merge and d_cid in cseeds:
                continue

            # Adjust all the amounts deposited to the cluster by the sender
            outs = {(t, o, 1) for o in t.outputs if entity.has_addr(o.address)}
            d_cids[d_cid] = {
                    'outs': outs,
                    'addrs_sending': {
                            f"{d_cid}:{bfe.addr_to_string(a)}"
                            for a in input_addrs
                        },
                    'addrs_receiving': {
                            f"{cid}:{bfe.addr_to_string(o.address)}"
                            for (t, o, p) in outs
                        },
                }

        for d_cid, d in d_cids.items():
            d_clusters.add(d_cid)
            e_clusters = update_outs(d_cid, d, e_clusters, height, True)

    # Forward

    for t in entity.input_txes:

        # Some very rare txes may have cero inputs, like this one:
        # a842b87403e6d6ca1a9ea39b16d496ebeb6ab15b83acb619cc10daba08114029
        if t.input_value == 0:
            logging.warning(f"Empty tx: {t.hash}\tinputs: {t.inputs.to_list()}\t{t.input_value} BTCs")
            continue

        w_cids = defaultdict(set)
        coinjoin = blocksci.heuristics.is_coinjoin(t)
        input_addresses = set()

        # There may be some senders from different clusters when using art_c
        if coinjoin or art_c:

            if coinjoin:
                entity.update_coinjoin_txes_in({str(t.hash)})
                logging.debug(f"COINJOIN WITHDRAWAL: {t.hash}")

            # Identify inputs of this cluster, and the total amount sent
            sender = defaultdict(int)
            total_deposited = 0
            for i in t.inputs.to_list():
                w_cid = get_cluster(i.address, art_c, cm)
                w_value = i.value
                sender[w_cid] += w_value
                total_deposited += w_value
                if w_cid == cid:
                    input_addresses.add(i.address)

            try:
                # Replace the amount sent with the contribution percentage
                sender = {k: v/total_deposited for k, v in sender.items()}
            except ZeroDivisionError as ex:
                logging.error(f"ZeroDivisionError in a withdrawal")
                logging.error(f"tx: {t.hash}\tinputs: {t.inputs.to_list()}\tsender: {sender}\ttotal_deposited: {total_deposited}")
                raise ex

        else:
            input_addresses.update(t.inputs.address.to_list())

        for o in t.outputs.to_list():
            output_address = o.address
            w_cid = get_cluster(output_address, art_c, cm)

            # Avoid reporting values withdrawn to the same cluster
            if w_cid == cid:
                continue

            # Avoid reporting values withdrawn to the seed clusters
            elif merge and w_cid in cseeds:
                continue

            # Adjust the amounts sent proportionally with respect to the
            # contribution of this cluster
            p = sender[cid] if coinjoin or art_c else 1

            # Values being withdrawn by the cluster
            outs = (t, o, p)

            if w_cid not in w_cids:
                w_cids[w_cid] = {
                        'outs': set(),
                        'addrs_receiving': set(),
                        'addrs_sending': set()
                    }

            saddr_r = f"{w_cid}:{bfe.addr_to_string(output_address)}"
            w_cids[w_cid]['outs'].add(outs)
            w_cids[w_cid]['addrs_receiving'].add(saddr_r)
            w_cids[w_cid]['addrs_sending'].update([
                    f"{cid}:{bfe.addr_to_string(a)}" for a in input_addresses
                ])

        for w_cid, d in w_cids.items():
            w_clusters.add(w_cid)
            e_clusters = update_outs(w_cid, d, e_clusters, height, False)

    entity.d_clusters = d_clusters
    entity.w_clusters = w_clusters
    return entity, e_clusters

def get_cluster(address, art_c, cm):
    cid = None
    if art_c:
        cid = bfe.art_cluster_with_address(art_c, address)
    # cid is not found in art_c
    if cid is None:
        # if this is a no-clustering exploration, the MI-clustering doesn't
        # matter and the cid could be anything but -1
        if cm is None:
            cid = 0
        else:
            cid = cm.cluster_with_address(address).index
    return cid

def update_outs(cid, d, e_clusters, height, deposit=True):
    global cm
    utcdate = datetime.utcfromtimestamp

    if cid not in e_clusters:

        #TODO use a different value than -1
        if cid == -1:
            cowner, ctags, csize = 'coinbase_txes', '', 1
        else:
            if cm is None:
                cowner, ctags, csize = '', '', 1
            else:
                cowner, ctags, csize = search_tag(cid=cid, height=height)

        if csize == -1:
            c = cm.clusters()[cid]
            csize = c.addresses.size

        e_clusters[cid] = {
                    'cluster': cid,
                    'owner': cowner,
                    'ctags': ctags,
                    'csize': csize,
                    'sent_outs': set(),
                    'sent_to_seed': {
                        'addrs_sending': set(),
                        'addrs_receiving': set()
                    },
                    'recv_outs': set(),
                    'recv_from_seed': {
                        'addrs_sending': set(),
                        'addrs_receiving': set()
                    }
                }

    outs = {(
        str(t.hash),
        t.fee,
        t.block.timestamp,
        o.index,
        o.value*p,
        (o.value*p/1e8)*price.get_price(str(utcdate(t.block.timestamp))[:10]),
        bfe.addr_to_string(o.address)
        ) for (t, o, p) in d['outs']
    }
    addr_s = d['addrs_sending']
    addr_r = d['addrs_receiving']
    key_outs = 'sent_outs' if deposit else 'recv_outs'
    key_addrs = 'sent_to_seed' if deposit else 'recv_from_seed'
    e_clusters[cid][key_outs].update(outs)
    e_clusters[cid][key_addrs]['addrs_sending'].update(addr_s)
    e_clusters[cid][key_addrs]['addrs_receiving'].update(addr_r)

    return e_clusters

def to_json(entity, clusters, s_txes, odir):
    count = 0
    rows = []
    sent_total_value = 0
    sent_total_value_usd = 0
    recv_total_value = 0
    recv_total_value_usd = 0
    sent_all_txes = {}
    recv_all_txes = {}
    all_ts = set()
    for c, d in clusters.items():

#        logging.debug(f"Cluster: {c}, sent outs: {len(d['sent_outs'])}")
#        logging.debug(f"Cluster: {c}, recv outs: {len(d['recv_outs'])}")
        # Sum values sent from this cluster
        sent_value = 0
        sent_value_usd = 0
        sent_txes = {}
        sent_ts = set()
        for (h, f, t, i, v, u, a) in d['sent_outs']:
            sent_value += v / 1e8
            sent_value_usd += u
            sent_txes[str(h)] = f / 1e8
            sent_ts.add(t)
        sent_total_value += sent_value
        sent_total_value_usd += sent_value_usd
        sent_all_txes.update(sent_txes)
        if sent_ts:
            sent_ts = sorted(sent_ts)
            first_ts = str(datetime.utcfromtimestamp(sent_ts[0]))
            last_ts = str(datetime.utcfromtimestamp(sent_ts[-1]))
        else:
            first_ts = ''
            last_ts = ''
        all_txes = list(sent_txes.keys())
        all_ts.update(sent_ts)
        d['sent_to_seed']['value'] = round(sent_value, 8)
        d['sent_to_seed']['value_usd'] = round(sent_value_usd, 2)
        d['sent_to_seed']['fee'] = round(sum(sent_txes.values()), 8)
        d['sent_to_seed']['txes'] = len(sent_txes.keys())
        d['sent_to_seed']['txes_list'] = all_txes if s_txes else all_txes[:3]
        d['sent_to_seed']['first_tx_ts'] = first_ts
        d['sent_to_seed']['last_tx_ts'] = last_ts

        # Sum values received by this cluster
        recv_value = 0
        recv_value_usd = 0
        recv_txes = {}
        recv_ts = set()
        for (h, f, t, i, v, u, a) in d['recv_outs']:
            recv_value += v / 1e8
            recv_value_usd += u
            recv_txes[str(h)] = f / 1e8
            recv_ts.add(t)
        recv_total_value += recv_value
        recv_total_value_usd += recv_value_usd
        recv_all_txes.update(recv_txes)
        if recv_ts:
            recv_ts = sorted(recv_ts)
            first_ts = str(datetime.utcfromtimestamp(recv_ts[0]))
            last_ts = str(datetime.utcfromtimestamp(recv_ts[-1]))
        else:
            first_ts = ''
            last_ts = ''
        all_txes = list(recv_txes.keys())
        all_ts.update(recv_ts)
        d['recv_from_seed']['value'] = round(recv_value, 8)
        d['recv_from_seed']['value_usd'] = round(recv_value_usd, 2)
        d['recv_from_seed']['fee'] = round(sum(recv_txes.values()), 8)
        d['recv_from_seed']['txes'] = len(recv_txes.keys())
        d['recv_from_seed']['txes_list'] = all_txes if s_txes else all_txes[:3]
        d['recv_from_seed']['first_tx_ts'] = first_ts
        d['recv_from_seed']['last_tx_ts'] = last_ts

        d['total'] = round(sent_value + recv_value, 8)

    sent_total_fee = sum(sent_all_txes.values())
    sent_all_txes = len(sent_all_txes)
    recv_total_fee = sum(recv_all_txes.values())
    recv_all_txes = len(recv_all_txes)
    balance = round(sent_total_value-recv_total_value-recv_total_fee, 8)
    all_ts = sorted(all_ts)
    first_ts = str(datetime.utcfromtimestamp(all_ts[0])) if all_ts else ''
    last_ts = str(datetime.utcfromtimestamp(all_ts[-1])) if all_ts else ''

    jr = {
            'seed_cluster': entity.cid,
            'owner': entity.owner,
            'ctags': entity.ctags,
            'csize': entity.size,
            'service': entity.service,
            'ec-prob': entity.prob,
            'balance': balance,
            'first_tx_ts': first_ts,
            'last_tx_ts': last_ts,
            'deposits_total_value': round(sent_total_value, 8),
            'deposits_total_value_usd': round(sent_total_value_usd, 2),
##            'deposits_total_fees': round(sent_total_fee, 8),
            'deposits_total_txes': sent_all_txes,
            'deposits_total_coinjoin_txes': entity.coinjoin_txes_out_count,
            'withdrawals_total_value': round(recv_total_value, 8),
            'withdrawals_total_value_usd': round(recv_total_value_usd, 2),
            'withdrawals_total_fees': round(recv_total_fee, 8),
            'withdrawals_total_txes': recv_all_txes,
            'withdrawals_total_coinjoin_txes': entity.coinjoin_txes_in_count,
            'total_clusters': len(clusters),
            'clusters': []
        }
    s = sorted(clusters.items(), key=lambda x: x[1]['total'], reverse=True)
    for c, d in s:
        d.pop('sent_outs')
        d.pop('recv_outs')
        sts = d['sent_to_seed']
        d['sent_to_seed']['n_addrs_sending'] = len(sts['addrs_sending'])
        d['sent_to_seed']['addrs_sending'] = sorted(sts['addrs_sending'])
        d['sent_to_seed']['n_addrs_receiving'] = len(sts['addrs_receiving'])
        d['sent_to_seed']['addrs_receiving'] = sorted(sts['addrs_receiving'])
        rfs = d['recv_from_seed']
        d['recv_from_seed']['n_addrs_sending'] = len(rfs['addrs_sending'])
        d['recv_from_seed']['addrs_sending'] = sorted(rfs['addrs_sending'])
        d['recv_from_seed']['n_addrs_receiving'] = len(rfs['addrs_receiving'])
        d['recv_from_seed']['addrs_receiving'] = sorted(rfs['addrs_receiving'])
        jr['clusters'].append(d)
        count += 1

    ofile = os.path.join(odir, f"{entity.cid}_explore.json")
    with open(ofile, 'w') as f:
        f.write(json.dumps(jr) + '\n')

    summary(jr)

    return clusters

def summary(jr):
    to_seed = []
    from_seed = []
    for c in jr['clusters']:
        cid = c['cluster']
        owner = c['owner']
        size = c['csize']
        value = c['sent_to_seed']['value']
        if value:
            addrs_send = c['sent_to_seed']['n_addrs_sending']
            addrs_recv = c['sent_to_seed']['n_addrs_receiving']
            to_seed.append([value, addrs_recv, addrs_send, cid, owner, size])
        value = c['recv_from_seed']['value']
        if value:
            addrs_send = c['recv_from_seed']['n_addrs_sending']
            addrs_recv = c['recv_from_seed']['n_addrs_receiving']
            from_seed.append([value, addrs_recv, addrs_send, cid, owner, size])

    msg = (
            f"\nSummary of cluster {jr['seed_cluster']} (size {jr['csize']:,})"
            f"\nTotal deposited: {jr['deposits_total_value']:,} BTC "
            f"({jr['deposits_total_value_usd']:,} USD)"
        )

    msg += f"\n\nDeposits:"
    for row in to_seed:
        value, addrs_recv, addrs_send, cid, owner, size = row
        msg += (
                f"\n{value:,} BTC received by {addrs_recv:,} address"
                f"{'es' if addrs_recv > 1 else ''} from "
                f"{addrs_send:,} address{'es' if addrs_send > 1 else ''} "
                f"of cluster {cid}:{owner}"
            )

    msg += f"\n\nWithdrawals: "
    for row in from_seed:
        value, addrs_recv, addrs_send, cid, owner, size = row
        msg += (
                f"\n{value:,} BTC sent by {addrs_send:,} address"
                f"{'es' if addrs_send > 1 else ''} to "
                f"{addrs_recv:,} address{'es' if addrs_recv > 1 else ''} "
                f"of cluster {cid}:{owner}"
            )
    logging.info(msg)

def to_graph(entity, e_clusters):
    G = nx.DiGraph()
    G.add_node(entity.cid, owner=entity.owner, ctags=entity.ctags,
            csize=entity.size, seed='1')
    G.nodes[entity.cid]['graphics'] = {'fill': "#7800ff"}

    for c, d in e_clusters.items():
        total_sent = d['sent_to_seed']['value']
        total_sent_usd = d['sent_to_seed']['value_usd']
        total_recv = d['recv_from_seed']['value']
        total_recv_usd = d['recv_from_seed']['value_usd']
        G.add_node(c, owner=d['owner'], ctags=d['ctags'], csize=d['csize'])
        G.nodes[entity.cid]['graphics'] = {'fill': "#00ff00"}

        if total_sent:
            f_ts = d['sent_to_seed']['first_tx_ts']
            l_ts = d['sent_to_seed']['last_tx_ts']
            ntxes = d['sent_to_seed']['txes']
            G.add_edge(c, entity.cid, weight=total_sent, usd=total_sent_usd,
                    ntxes=ntxes, firstts=f_ts, lastts=l_ts)

        if total_recv:
            f_ts = d['recv_from_seed']['first_tx_ts']
            l_ts = d['recv_from_seed']['last_tx_ts']
            ntxes = d['recv_from_seed']['txes']
            G.add_edge(entity.cid, c, weight=total_recv, usd=total_recv_usd,
                    ntxes=ntxes, firstts=f_ts, lastts=l_ts)

    return G

def merge_clusters(merged_clusters, e_clusters):

    for c in e_clusters:

        if c in merged_clusters:

            e = e_clusters[c]['sent_to_seed']
            m = merged_clusters[c]['sent_to_seed']
            m['addrs_sending'].update(e['addrs_sending'])
            m['addrs_receiving'].update(e['addrs_receiving'])
            merged_clusters[c]['sent_outs'].update(e_clusters[c]['sent_outs'])

            e = e_clusters[c]['recv_from_seed']
            m = merged_clusters[c]['recv_from_seed']
            m['addrs_sending'].update(e['addrs_sending'])
            m['addrs_receiving'].update(e['addrs_receiving'])
            merged_clusters[c]['recv_outs'].update(e_clusters[c]['recv_outs'])

        else:

            merged_clusters[c] = copy.deepcopy(e_clusters[c])

    return merged_clusters

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


#def estimate_all(height, clusters, art_c=None):
#    global_total = 0
#
#    for cid in clusters:
#        if art_c:
#            total = 0
#            for addr in art_c[cid]['addrs']:
#                total += sum(addr.outputs.value) / 1e8
#        else:
#            c = cm.clusters()[cid]
#            total = sum(c.outputs().value) / 1e8
#        global_total += total
#        logging.info(f"Cluster: {cid}\tTotal received: {total:.8f}")
#
#    logging.info(f"Total received: {global_total:.8f}")


def main(args):
    global chain, cm
    chain, cm = bfe.build_load_blocksci(args.blocksci, args.height)

    if args.artificial:
        art_c = bfe.build_art_clusters(chain, args.artificial, args.height,
                    ignore_wd=False)
        if not art_c:
            msg = f"File {args.artificial} is not a valid mapping file."
            logging.error(msg)
            return
    else:
        art_c = None

    if args.clusters_file:
        fcids = pd.read_csv(args.clusters_file, sep='\t').cid.unique()
        fcids = [int(cid) for cid in fcids]
        args.clusters = args.clusters.extend(fcids) if args.clusters else fcids

    if args.classifier:
        classifier = load(args.classifier)
    else:
        classifier = None
        logging.warning('Exchange classifier not set.')

    explore_clusters(args.height, args.merge, args.clusters, art_c, args.txes,
            args.output, args.operation, ignore_wd=False, save=True,
            classifier=classifier)


if __name__ == '__main__':
    desc = "Explore deposits and withdrawals of a cluster"
    version = '1.0'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-D', '--blocksci', type=str,
            help='Blocksci config file')
    parser.add_argument('-H', '--height', type=int, default=769900,
            help='Max block height')
    parser.add_argument('-c', '--clusters', action='append', type=int,
            help='Cluster IDs to be explored')
    parser.add_argument('-F', '--clusters-file', default=None,
            help='File with a list of cluster IDs to be explored')
    parser.add_argument('-t', '--txes', action="store_true",
            help='Store all transactions found into the output file')
    parser.add_argument('-M', '--merge', action='store_true',
            help='Assume that the given cluster IDs belong to the same entity')
    parser.add_argument('-o', '--operation', type=str, default=None,
            help='A tag for the owner of the explored clusters')
    parser.add_argument('-A', '--artificial', action='store', default=None,
            help='Use this file to manually map addresses to new clusters')
    parser.add_argument('-O', '--output', type=str, default='./',
            help='Save all outputs into this folder')
    parser.add_argument('-C', '--classifier', type=str, default=None,
            help='Path to the exchange classifier')
    parser.add_argument('-v', '--version', action='version', version=version)

    args = parser.parse_args()

    e = set()
    if not (args.blocksci and os.path.isfile(args.blocksci)):
        e.add("Blocksci data not found.")
    if args.clusters_file and not os.path.isfile(args.clusters_file):
        e.add("A valid file should be given to the --clusters-file parameter.")
    if args.artificial and not os.path.isfile(args.artificial):
        e.add("A valid file should be given to the --artificial parameter.")
    if args.output:
        if not os.path.isdir(args.output):
            e.add("A valid folder should be given to the --output parameter.")
        elif not os.access(args.output, os.W_OK):
            e.add(f"Folder without write permissions: {args.output}")
    if args.classifier and not os.path.isfile(args.classifier):
        e.add("Exchange classifier file does not exist.")
    if e:
        parser.error("\n".join(e))

    op = 'entity_exploration'
    oplist = ','.join([str(c) for c in (args.clusters or [])])
    op = args.operation or oplist or op
    flog = os.path.join(args.output, f"{op}.log")
    logging.basicConfig(filename=flog, level=logging.DEBUG)
    logging.debug(f"Made with version {version}")
    logging.debug(f"- Added exchange classifier results to seeds")
    logging.debug(f"- Added summary per cluster explored")
    logging.debug(f"- Added first/last_tx_ts to seed cluster; no deposit fees")
    logging.debug(f"- Added object Entity")
    logging.debug(f"- Added ignore_wd parameter to art_c")
    logging.debug(f"{args}")

    main(args)
