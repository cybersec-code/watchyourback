#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import logging
import blocksci
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
try:
    from lib import blockchain_feature_extraction as bfe
except ModuleNotFoundError:
    import blockchain_feature_extraction as bfe

# Set the default tags_dir
file_path = os.path.abspath(os.path.dirname(__file__))
tags_dir = os.path.join(file_path, '..', '..', 'data', 'tagging')

dbs = {}
tcs = {}

def build(chain, cm, ticker='btc', height=0):
    build_address_tags(ticker)
    clean_address_tags(ticker, height)
    build_cluster_tags(chain, cm, ticker, height)
    clean_cluster_tags(cm, ticker, height)
    produce_owner_user_tags(ticker, height)

def build_address_tags(ticker='btc'):
    '''
    Collect all source tag files under tags_dir and join them together into a
    file using the same fields. Exclude special files.
    :param str ticker: Blockchain to use, default=btc
    '''
    global dbs, tcs
    names = ['address', 'ticker', 'category', 'tag', 'subtype', 'url']
    fn = os.path.join(tags_dir, f"{ticker}_all.tags")

    if os.path.exists(fn):
        logging.warning(f"File {fn} exists already, it will be used as it is.")
        db = pd.read_csv(fn, names=names, na_filter=False)
        dbs[ticker] = db
        n = db.shape[0]
        u = db.address.unique().size
        logging.info(f"Loading {n} {ticker} tags, for {u} unique addresses.")
        return

    # Read all sources (*.tags)
    db = pd.DataFrame()
    # Old special files
    tag_f = ["_all", "_all-mi", "_all_clean", "_all-mi_clean"]
    # New special files
    tag_f.extend(["_solved", "_solved-mi", "_unsolved", "_unsolved-mi"])
    for path in Path(tags_dir).rglob('*.tags'):
        logging.debug(f"Reading tags file: {path.name}")
        if any([path.name.endswith(f"{p}.tags") for p in tag_f]):
            logging.debug(f"Omiting file: {path.name}")
            continue
        f = path.resolve()
        new = pd.read_csv(f, names=names, na_filter=False, comment='#')
        db = pd.concat([db, new], ignore_index=True)

    # Use resolv.csv file to resolv conflicts, if any
    f = os.path.join(tags_dir, f"{ticker}_resolv.csv")
    if os.path.isfile(f):
        logging.debug(f"Resolving conflicts with file {f}")
        resolv = pd.read_csv(f, names=names, na_filter=False, comment='#')
        idx = db[db.address.isin(resolv.address)]
        db.drop(index=idx.index, inplace=True)
        db = pd.concat([db, resolv], ignore_index=True)

    db.drop_duplicates(inplace=True)
    # keep tags of this ticker, drop the rest
    nonidx = db[db.ticker!=ticker].index
    logging.info(f"Dropping {nonidx.shape[0]} tags not from {ticker}.")
    db.drop(nonidx, inplace=True)
    db.to_csv(fn, index=None, header=None)

    n = db.shape[0]
    u = db.address.unique().size
    logging.info(f"Built {n} {ticker} tags, for {u} unique addresses.")
    dbs[ticker] = db

def build_cluster_tags(chain=None, cm=None, ticker='btc', height=0):
    global dbs, tcs
    db = dbs[ticker]
    names = ['cid', 'size', 'tags']
    c_height = f"_h{height+1}" if height else ''
    name = f"{ticker}{c_height}_all-mi.tags"
    fn = os.path.join(tags_dir, name)

    # Build a dict with cluster tags by iterating all tagged addresses
    dbc = list()
    fields = ['ticker', 'category', 'tag', 'subtype']
    clusters = {}
    for saddr, group in db.groupby('address'):
        k = bfe.addr_from_string(saddr, chain, height)
        # Ignore addrs without txes
        if k is None:
            continue

        t = group[fields]
        v = t.agg('='.join, axis=1).unique()
        c = cm.cluster_with_address(k)
        cid = c.index
        t['cid'] = cid
        t['address'] = saddr
        dbc.extend(t.to_dict('records'))

        if cid in clusters:
            clusters[cid]['tags'].update(v)
        else:
            csize = c.address_count()
            clusters[cid] = {'size': csize, 'tags': set(v)}

    # Update global dict of address tags
    columns = ['ticker', 'cid', 'address', 'category', 'tag', 'subtype']
    db = pd.DataFrame(data=dbc, columns=columns)
    dbs[ticker] = db

    # Use *resolv_clusters.csv file to resolv conflicts
    rc = os.path.join(tags_dir, f"{ticker}_resolv_clusters.csv")
    if os.path.isfile(rc):
        logging.debug(f"Resolving cluster conflicts with file {rc}")
        names_rc = ['leader', 'ticker', 'service', 'alias']
        resolv = pd.read_csv(rc, names=names_rc, na_filter=False, comment='#')
        for i, e in resolv.iterrows():
            leader, tckr, serv, alias = e
            # We search for the CID of the leader
            try:
                cid = db[db.address==leader].cid.iloc[0]
            except IndexError:
                logging.warning(f"Address not found in database: {leader}")
                continue
            # We take the first alias, which should be the most relevant
            first_alias = alias.split(';')[0]
            clusters[cid]['tags'] = set([f"{tckr}={serv}={first_alias}="])

    tc = {cid: {'tags': ';'.join(sorted(c['tags'])), 'size': c['size']}
            for cid, c in clusters.items()}
    data = [(c, d['size'], d['tags']) for c, d in tc.items()]
    df = pd.DataFrame(data=data, columns=names)
    df.to_csv(fn, index=None, header=None)

    logging.info(f"Built {len(tc)} {ticker} clusters with tags.")
    tcs[ticker] = tc

def clean_address_tags(ticker='btc', height=0):
    global dbs #, aliases
    db = dbs[ticker]
    height = f"_h{height+1}" if height else ''
    uname = f"{ticker}{height}_unsolved.tags"
    names = ['ticker', 'address', 'category', 'tag', 'subtype']
    fields = ['ticker', 'category', 'tag', 'subtype']
    ufn = os.path.join(tags_dir, uname)
    solved = list()
    unsolved = list()

    # Remove pending tags to not generate conflicts with miscabuse
    pending = db[db.category=='pending'].address
    miscabuse = db[db.category=='miscabuse'].address
    idx = db[(db.address.isin(pending))&(db.address.isin(miscabuse))
            &(db.category=='pending')].index

    # Solve aaron-smith sextortion campaign
    aaronsmith = db[db.tag=='aaron-smith'].address
    sextortion = db[db.tag=='sextortion-spam'].address
    idx = idx.append(db[(db.address.isin(aaronsmith))
            &(db.address.isin(sextortion))&(db.tag=='sextortion-spam')].index)

    # Solve bitcoinwhoswho-report
    whoswho = db[db.tag=='bitcoinwhoswho-report'].address
    idx = idx.append(db[(db.address.isin(whoswho))&(db.address.isin(sextortion))
            &(db.tag=='bitcoinwhoswho-report')].index)

    # Solve OFAC listed 
    ofactag = 'asset-listed-under-us-treasury-ofac-sanctions-list'
    ofac_new = db[db.tag==ofactag].address
    ofac_old = db[db.tag=='ofac-sanctions-list'].address
    idx = idx.append(db[(db.address.isin(ofac_new))&(db.address.isin(ofac_old))
            &(db.tag=='ofac-sanctions-list')].index)

    # Solve hydra tormarket and OFAC
    hydra = db[db.tag=='hydra-market'].address
    idx = idx.append(db[(db.address.isin(hydra))&(db.address.isin(ofac_new))
            &(db.tag==ofactag)].index)

    # Solve apt29 and OFAC
    apt29 = db[(db.category=='state-sponsored')&(db.tag=='apt29')].address
    idx = idx.append(db[(db.address.isin(apt29))&(db.address.isin(ofac_new))
            &(db.tag==ofactag)].index)

    db.drop(idx, inplace=True)

    for saddr, group in db.groupby('address'):
        tags = set(group[fields].apply('='.join, axis=1).unique())
        otags = tags.copy()

        # There is just one tag for this address
        if len(tags) == 1:
            update_addr_tags(solved, saddr, tags.pop())
            continue

        # Try to solve cases with more than one tag
        uniq_tags = set()
        while tags:
            # get the category and tag of the longest tag
            first_tag = sorted(tags, key=len, reverse=True)[0].split('=')
            tckr, cat, tag, subt = first_tag
            tag = remove_tld(tag)
            # remove all tags similar to first_tag
            for t in [x for x in tags if x.startswith(f"{tckr}={cat}={tag}")]:
                tags.remove(t)
            # include first_tag in the unique set, excluding miscabuse/pending
            if 'miscabuse' not in first_tag and 'pending' not in first_tag:
                uniq_tags.add('='.join(first_tag))

        # If there is only one tag left, assign it to the address
        if len(uniq_tags) == 1:
            update_addr_tags(solved, saddr, uniq_tags.pop())
            continue
        # More than one different tag here
        elif len(uniq_tags) > 1:
            # TODO we need to check the onlinewallet case, if the same address
            # is tagged as service (e.g. exchange) and other (e.g. ransomware)
            logging.info(f"Unable to clean address {saddr}: {uniq_tags}")
            for t in uniq_tags:
                update_addr_tags(unsolved, saddr, t)
        # All tags were bitcoinabuse or pending
        else:
            msg = f"Unable to clean address (miscabuse) {saddr}: {otags}"
            logging.info(msg)
            for t in otags:
                update_addr_tags(unsolved, saddr, t)

    unsolved = pd.DataFrame(data=unsolved, columns=names)
    unsolved.to_csv(ufn, index=None, header=None)
    dbs[ticker] = pd.DataFrame(data=solved, columns=names)

def clean_cluster_tags(cm, ticker='btc', height=0):
    global tcs
    tc = tcs[ticker]
    untc = {}
    unique, miscabuse, solved, unsolved = 0, 0, 0, 0

    for cid, d in tc.items():
        tags_list = d['tags']
        tags = set(tags_list.split(';'))

        if len(tags) < 2:
            tc[cid]['owner'] = cluster_owner(d['tags'])
            unique += 1
            continue

        uniq_tags = set()
        while tags:
            # get the category and tag of the longest tag
            first_tag = sorted(tags, key=len, reverse=True)[0].split('=')
            [tckr, cat, tag, subt] = first_tag
            tag = remove_tld(tag)
            # remove similar tags
            for t in [x for x in tags if x.startswith(f"{tckr}={cat}={tag}")]:
                tags.remove(t)
            if 'miscabuse' not in first_tag and 'pending' not in first_tag:
                uniq_tags.add('='.join(first_tag))

        if uniq_tags:
            tc[cid]['tags'] = ';'.join(sorted(uniq_tags))
            owner = cluster_owner(tc[cid]['tags'])
            tc[cid]['owner'] = owner
            # more than one different tag here, owner is unsolved
            if len(uniq_tags) > 1 and (owner in ['unknown', 'more-than-one']):
                c = cm.clusters()[cid]
                s = tc[cid]['size']
                info = f"{cid}\tsize: {s}\ttags: {tc[cid]}"
                logging.info(f"Unable to clean cluster {info}")
                unsolved += 1
                untc[cid] = tc[cid]
            # just one tag left or owner is solved
            else:
                solved += 1
        # All tags were bitcoinabuse with different tags/+ pending
        else:
            tc[cid]['owner'] = 'miscabuse='
            miscabuse += 1
            msg = f"Unable to clean cluster (miscabuse) {cid}: {tags_list}"
            logging.info(msg)
            untc[cid] = tc[cid]

    msg = f"Clusters with only tag: {unique}; solved: {solved};"
    msg += f" unsolved: {unsolved}; miscabuse: {miscabuse};"
    logging.info(msg)

    # Save solved clusters
    save_cluster_dataset(tc, ticker, height)

    # Save unsolved clusters
    save_cluster_dataset(untc, ticker, height, solved=False)

    tcs[ticker] = tc

def save_cluster_dataset(tc, ticker='btc', height=0, solved=True):
    c_height = f"_h{height+1}" if height else ''

    if solved:
        names = ['cid', 'size', 'owner', 'tags']
        name = f"{ticker}{c_height}_solved-mi.tags"
        data = [(c, d['size'], d['owner'], d['tags']) for c, d in tc.items()]
    else:
        names = ['cid', 'size', 'tags']
        name = f"{ticker}{c_height}_unsolved-mi.tags"
        data = [(c, d['size'], d['tags']) for c, d in tc.items()]

    fn = os.path.join(tags_dir, name)
    df = pd.DataFrame(data=data, columns=names)
    df.to_csv(fn, index=None, header=None)

def cluster_owner(ctags):
    '''
    Resolv the owner of a cluster with several ctags. In some cases there can
    be two different, similar tags, or the same tag with different
    categories (e.g. service and mining). We try to merge similar tags when
    one or both of them are domain names, and we give priority to categories
    other than service.
    '''
    # If the cluster has only one tag, this is the owner
    stags = ctags.split(';')
    if len(stags) == 1:
        tckr, cat, tag, subt = ctags.split('=')
        return f"{cat}={tag}"
    # If it has more than one, we search for service tags
    owners = set()
    tags = set()
    possible_owner = ''
    for tag in stags:
        if is_service_ctags(tag):
            tckr, cat, tag, subt = tag.split('=')
            owners.add(f"{cat}={tag}")
            tags.add(remove_tld(tag))
            # We omit the generic category 'service' in favor of others
            if cat != 'service':
                possible_owner = f"{cat}={tag}"
    if len(owners) == 1:
        return owners.pop()
    elif len(owners) > 1:
        # If there is just one tag, it's probably the same service
        if len(tags) == 1:
            return possible_owner
        else:
            return 'more-than-one'
    else:
        return 'unknown'

def update_addr_tags(db, addr, t):
    tckr, cat, tag, subt = t.split('=')
    d = {'address': addr, 'ticker': tckr, 'category': cat, 'tag': tag,
         'subtype': subt}
    db.append(d)

def remove_tld(tag):
    """
    Remove everything after a dot in order to remove TLD of domain-like tags
    (e.g. kraken.com). Due these tags are for the same service (e.g. same addr
    or same cluster) we do not care much if we remove E2LD (e.g. .co.uk) or
    the complete domain (e.g. .netlify.com), we just want to compare tags.
    """
    try:
        tmptag = tag[:tag.index(".")]
    except ValueError:
        tmptag = tag
    return tmptag

def produce_owner_user_tags(ticker, height):
    global dbs, tcs
    db = dbs[ticker]
    tc = tcs[ticker]
    height = f"_h{height+1}" if height else ''
    name = f"{ticker}{height}_solved.tags"
    names = ['ticker', 'cid', 'address', 'owner', 'user']
    fn = os.path.join(tags_dir, name)
    solved = list()
    fields = ['category', 'tag', 'subtype']

    for cid, group in db.groupby('cid'):
        for i, e in group.iterrows():
            # TODO what to do with addrs in unsolved clusters?
            # We can use owner = user, or owner = '', or owner = 'unknown'
            o = tc[cid]['owner'] if cid in tc else 'unknown'
            u = '='.join(e[fields])
            a = e['address']
            t = e['ticker']
            d = {'ticker': t, 'cid': cid, 'address': a, 'owner': o, 'user': u}
            solved.append(d)

    db = pd.DataFrame(data=solved, columns=names)
    dbs[ticker] = db

    # Include BTC-imported tagged addresses
    c_height = f"_h{height}" if height else ''
    f = os.path.join(tags_dir, f"btc_to_{ticker}{c_height}.csv")
    if os.path.isfile(f):
        logging.debug(f"Including BTC-imported addresses in {f}")
        imported = pd.read_csv(f, names=names, na_filter=False, comment='#')
        db = solve_owner_conflicts(db, imported, ticker, height)
        dbs[ticker] = db

    dbs[ticker].to_csv(fn, index=None, header=None)

def solve_owner_conflicts(db, imported, ticker, height):
    global tcs
    tc = tcs[ticker]

    # Identify owner conflicts by checking the imported CIDs
    update_tc = False
    for cid, gimported in imported.groupby('cid'):
        prev_idx = db[db.cid==cid].index
        new_idx = gimported.index
        # There were no previous tags for this CID
        if not any(prev_idx):
            czise = 0 # cm.clusters(cid).addresses.size
            owner = gimported.owner.unique().item()
            ctags = gimported.user.unique().item()
            tc[cid] = {'size': csize, 'owner': owner, 'tags': ctags}
            continue
        # There should exist just one owner per CID
        prev_owner = db.loc[prev_idx].owner.unique().item()
        new_owner = gimported.owner.unique().item()
        # Add the new user-tag to the cluster
        tc[cid]['tags'] = add_cluster_tags(gimported, tc[cid])
        # If both owners are equal we just continue
        if prev_owner == new_owner:
            continue
        if prev_owner == 'unknown' or new_owner == 'more-than-one':
            db.loc[prev_idx, ['owner']] = new_owner
            tc[cid]['owner'] = new_owner
        elif prev_owner == 'more-than-one' or new_owner == 'unknown':
            imported.loc[new_idx, ['owner']] = prev_owner
        else: # Both prev and new owner were solved and are different
            old_is_serv = is_service_owner(prev_owner)
            new_is_serv = is_service_owner(new_owner)
            if old_is_serv and new_is_serv:
                imported.loc[new_idx, ['owner']] = 'more-than-one'
                db.loc[prev_idx, ['owner']] = 'more-than-one'
                tc[cid]['owner'] = 'more-than-one'
            elif new_is_serv:
                db.loc[prev_idx, ['owner']] = new_owner
                tc[cid]['owner'] = new_owner
            elif old_is_serv:
                imported.loc[new_idx, ['owner']] = prev_owner
            else:
                imported.loc[new_idx, ['owner']] = 'unknown'
                db.loc[prev_idx, ['owner']] = 'unknown'
                tc[cid]['owner'] = 'unknown'

    db = pd.concat([db, imported], ignore_index=True)

    # Update the owner in already solved clusters
    save_cluster_dataset(tc, ticker, height)
    tcs[ticker] = tc

    # TODO Update unsolved clusters
#    save_cluster_dataset(untc, ticker, height, solved=False)

    return db.drop_duplicates()

def add_cluster_tags(gimported, c):
    ctags = gimported[['ticker', 'user']].agg('='.join, axis=1).values
    return ';'.join(sorted(set(c['tags'].split(';') + list(ctags))))

def load_address_tags(ticker='btc', height=0):
    global dbs
    names = ['ticker', 'cid', 'address', 'owner', 'user']
    c_height = f"_h{height+1}" if height else ''
    name = f"{ticker}{c_height}_solved.tags"
    fn = os.path.join(tags_dir, name)
    db = pd.read_csv(fn, names=names, na_filter=False)
    n = db.shape[0]
    u = db.address.unique().size
    logging.info(f"Loaded {n} {ticker} tags, for {u} unique addresses.")
    dbs[ticker] = db
    return dbs

def load_cluster_tags(ticker='btc', height=0):
    global dbs
    global tcs
    if ticker not in dbs:
        dbs = load_address_tags(ticker=ticker, height=height)
    db = dbs[ticker]
    names = ['cid', 'size', 'owner', 'tags']
    c_height = f"_h{height+1}" if height else ''
    name = f"{ticker}{c_height}_solved-mi.tags"
    fn = os.path.join(tags_dir, name)

    df = pd.read_csv(fn, names=names, na_filter=False)
    tc = df.set_index('cid').T.to_dict()

    logging.info(f"Loaded {len(tc)} {ticker} clusters with tags.")
    tcs[ticker] = tc

    return tcs

def search_tag(addr, ticker='btc', height=0):
    global dbs
    if ticker not in dbs:
        dbs = load_address_tags(ticker=ticker, height=height)
    db = dbs[ticker]
    saddr = addr if isinstance(addr, str) else bfe.addr_to_string(addr)
    tags = db[db.address==saddr][['owner', 'user']].values
    owner, user = tags[0] if tags.size > 0 else ('', '')
    return owner, user

def search_tag_by_cluster(addr=None, chain=None, cm=None, cid=None,
        ticker='btc', height=0):
    global tcs
    if ticker not in tcs:
        tcs = load_cluster_tags(ticker=ticker, height=height)
    tc = tcs[ticker]
    if cid is None:
        if isinstance(addr, str):
            addr = bfe.addr_from_string(addr, chain, height)
        cid = cm.cluster_with_address(addr).index
    if cid in tc:
        return tc[cid]['owner'], tc[cid]['tags'], tc[cid]['size']
    return '', '', -1

def is_service_ctags(ctags):
    '''Determine the type of service of this set of cluster-tags.'''
    if not ctags:
        return ''
    for stag in ctags.split(';'):
        [ticker, cat, tag, subt] = stag.split('=')
        if cat in ['exchange', 'onlinewallet', 'defi']:
            return 'exchange'
        elif cat in ['mixer', 'tormarket', 'gambling', 'mining', 'payment-processor', 'service']:
            return cat
    return ''

def is_service_tag(stag):
    '''Determine the type of service of this address tag.'''
    if not stag:
        return ''
    [cat, tag, subt] = stag.split('=')
    if cat in ['exchange', 'onlinewallet', 'defi']:
        return 'exchange'
    elif cat in ['mixer', 'tormarket', 'gambling', 'mining', 'payment-processor', 'service']:
        return cat
    return ''

def is_service_owner(owner):
    '''Determine if this owner-tag is a service.'''
    if not owner or owner == 'unknown':
        return ''
    [cat, tag] = owner.split('=')
    services = ['exchange', 'onlinewallet', 'defi', 'mixer', 'tormarket']
    services += ['gambling', 'mining', 'payment-processor', 'service']
    services += ['more-than-one']
    return cat if cat in services else ''


def main(args):

    # Load BlockSci parsed data
    chain, cm = bfe.build_load_blocksci(args.blocksci, args.height)

    if args.build:
        build(chain, cm, args.ticker, args.height)
        return


if __name__ == "__main__":
    version = "2.0.2"
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--ticker', choices=['btc', 'bch', 'ltc'],
            default='btc', help='Select blockchain to work with (default=btc)')
    parser.add_argument('-D', '--blocksci', dest='blocksci', default='',
            help='Path to the BlockSci config file')
    parser.add_argument('-B', '--build', action='store_true', default=False,
            help='Build address/cluster tag databases')
    parser.add_argument('-H', '--height', action='store', default=0, type=int,
            help='Blockchain height to analyze')
    parser.add_argument('-f', '--fork', action='store', default=0, type=int,
            help='Height of the fork from Bitcoin (e.g. for BCH is 478558)')
    parser.add_argument('-v', '--version', action='version', version=version)

    args = parser.parse_args()

    if not args.blocksci or not os.path.isfile(args.blocksci):
        parser.error(f"A valid BlockSci config file is needed.")

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Made with version {version}")
    logging.debug(f"- solve owner conflicts")
    logging.debug(f"- include BTC-imported tagged addresses")
    logging.debug(f"- fixes to clean tags")
    logging.debug(f"- refactor to define owner and user fields")
    logging.debug(f"- added lightning network tags")
    logging.debug(f"- added defi to services")
    logging.debug(f"- solve ctags using *resolv_clusters.csv file")
    logging.debug(f"- return ctags in clusters with more than one service tag")
    logging.debug(f"- introduced height to ctags")
    logging.debug(f"{args}")

    main(args)

