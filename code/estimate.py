import logging
import argparse, os
from lib import blockchain_feature_extraction as bfe
from lib.tags import search_tag_by_cluster as search_tag
from lib.tags import is_service_owner
from lib import ransomware
from lib import price_api as price
from lib import entity_exploration as eexp
from datetime import datetime, timezone
import pandas as pd

chain = None
cm_mi = None
cm_mica = None

## Height 785100
ec_ransomwhere = [142458720, 18487128]
ec_exchange_scams = [238098406]
ec_giveaway_scams = [87303754, 85864202, 660907037, 669095226, 84554767]

# TODO read this dinamically
ec = set(ec_ransomwhere + ec_exchange_scams + ec_giveaway_scams)

def expansion(seeds, clustering, tag_filter, height):
    global chain, cm_mi, cm_mica
    if not chain and (not cm_mi or not cm_mica):
        logging.error(f"Chain is not loaded. Exiting.")
        return

    clusters = {}
    for seed in seeds:
        addr = bfe.addr_from_string(seed, chain)
        if not addr:
            continue

        # Select a mi/mi+ca cluster
        if clustering == 'mi':
            c = cm_mi.cluster_with_address(addr)
        elif clustering == 'mica':
            c = cm_mica.cluster_with_address(addr)
        else:
            c = None

        cid = c.index if c else -1
        if cid in clusters:
            if cid == -1 or clusters[cid]['service']:
                clusters[cid]['addrs'].add(addr)
            continue

        clusters[cid] = {}

        # Filter service addresses
        if c and tag_filter:

            change = clustering == 'mica'
            cowner, ctags, csize = search_tag(cid=c.index, height=height, change=change)
            service = is_service_owner(cowner)
            if cid in ec:
                service = 'EC'

            clusters[cid]['service'] = service

            addrs = {addr} if service else set(c.addresses.to_list())
            clusters[cid]['addrs'] = addrs

        elif c:

            clusters[cid]['service'] = ''
            clusters[cid]['addrs'] = set(c.addresses.to_list())

        else:

            clusters[cid]['service'] = ''
            clusters[cid]['addrs'] = {addr}

    return clusters

def get_estimation_outs(clusters, dc, filter, height):
    outs = []

    # Apply a DD (Direct Deposits) estimation
    if not dc:

        for cid in clusters:
            for addr in clusters[cid]['addrs']:
                for o in addr.outputs.where(lambda t: t.block.height<=height):
                    a = bfe.addr_to_string(o.address)
                    outs.append([o.tx.fee, o.tx.block.timestamp, o.value, a])

    # Use DC filter in the estimation
    else:

        dictmap = {cid: clusters[cid]['addrs'] for cid in clusters}

        art_c = bfe.build_art_clusters_from_dict(chain, dictmap, height,
                ignore_wd=True)
        cl = dictmap.keys()

        # We skip withdrawals by default, we just need deposits
        merged_clusters = eexp.explore_clusters(height, True, cl, art_c,
                txes=True, output='', operation='', ignore_wd=True, save=False)

        for c in merged_clusters:
            for (h, f, t, i, v, u, a) in merged_clusters[c]['sent_outs']:
                outs.append((f, t, v, a))

        merged_clusters = None

    # Value/Time filters
    if filter == 'vf':
        filter_func = ransomware.is_cryptolocker_payment_vf
    elif filter == 'tf':
        filter_func = ransomware.is_cryptolocker_payment_tf
    elif filter == 'vtf':
        # This method consider the BTC amounts as well as the
        # converted USD amounts for filtering a CryptoLocker payment
        filter_func = ransomware.is_cryptolocker_payment_vtf
        # This method consider just the BTC amounts
        # filter_func = ransomware.is_cryptolocker_payment_btc
    else:
        filter_func = None


    addrs, dep, btc, usd = set(), 0, 0, 0
    faddrs, fdep, fbtc, fusd = set(), 0, 0, 0
    period, fperiod = set(), set()
    while outs:
        [f, t, v, a] = outs.pop()
        ts = datetime.utcfromtimestamp(t)
        ts = ts.replace(tzinfo=timezone.utc)
        usd_ex = price.get_price(str(ts)[:10])
        period.add(str(ts)[:10])

        addrs.add(a)
        dep += 1
        btc += v/1e8
        usd += (v/1e8 * usd_ex)

        if filter and filter_func(v, f, ts):
            faddrs.add(a)
            fdep += 1
            fbtc += v/1e8
            fusd += (v/1e8 * usd_ex)
            fperiod.add(str(ts)[:10])

    period = get_start_end_period(period)
    if filter:
        fperiod = get_start_end_period(fperiod)

    return (len(addrs), dep, period, btc, usd,
            len(faddrs), fdep, fperiod, fbtc, fusd)

def get_start_end_period(period):
    period = sorted(period)
    if len(period) > 1:
        return period[::len(period)-1]
    elif len(period) == 1:
        return [period[0], period[0]]
    else:
        return ['-', '-']

def calculate_profits(outs):
    btc_outs, usd_outs = [], []
    for (f, t, v, a) in outs:
        btc_outs.append(v/1e8)
        usd_ex = price.get_price(str(datetime.utcfromtimestamp(t))[:10])
        usd_outs.append((v/1e8)*usd_ex)

    return len(outs), sum(btc_outs), sum(usd_outs)

def estimation(seeds, dc, clust, filter, tag_filter, height):

    clusters = expansion(seeds, clust, tag_filter, height)

    if not clusters:
        logging.warning(f"The expanded set of seeds is empty.")
        return None

    return get_estimation_outs(clusters, dc, filter, height)

def estimation_name(args, filter=True):
    ow = '-OW' if args.tag_filter else ''
    exp = f"+{args.clustering.upper()}" if args.clustering else ''
    if filter:
        cf = f"-{args.filter.upper()}" if args.filter else ''
    else:
        cf = ''
    dc = '-DC' if args.doublecounting else ''
    return f"DD{ow}{exp}{cf}{dc}"

def load_chain(cnf, height, change=False):
    global chain, cm_mi, cm_mica
    if change:
        chain, cm_mica = bfe.build_load_blocksci(cnf, height, change=change)
    else:
        chain, cm_mi = bfe.build_load_blocksci(cnf, height)


def main(args):
    global chain, cm_mi, cm_mica

    if args.clustering == "mi":
        chain, cm_mi = bfe.build_load_blocksci(args.blocksci, args.height)
    elif args.clustering == "mica":
        _, cm_mica = bfe.build_load_blocksci(args.blocksci, args.height,
                change=True)
        chain, cm_mi = bfe.build_load_blocksci(args.blocksci, args.height)
    else:
        chain = bfe.load_chain(args.blocksci)

    if args.seeds_file:
        fseeds = pd.read_csv(args.seeds_file, names=['addr']).addr.unique()
        args.seeds = args.seeds.extend(fseeds) if args.seeds else fseeds

    e = estimation(args.seeds, args.doublecounting, args.clustering,
            args.filter, args.tag_filter, args.height)

    if e is None:
        logging.debug(f"Empty set of seeds.")
    else:
        addrs, dep, period, btc, usd, faddrs, fdep, fperiod, fbtc, fusd = e

    est = estimation_name(args, filter=False)
    logging.info(f"\n{est} Estimation:")
    logging.info(f"\tADDRS:\t{addrs:,}")
    logging.info(f"\tDEP:\t{dep:,}")
    logging.info(f"\tPERIOD:\t{period}")
    logging.info(f"\tBTC:\t{btc:,.8f}")
    logging.info(f"\tUSD:\t{usd:,.2f}")
    logging.debug(f"\n{addrs:,} & {dep:,} & {btc:,.4f} & {usd:,.0f}")

    if args.filter:
        est = estimation_name(args, filter=True)
        logging.info(f"\n{est} Estimation:")
        logging.info(f"\tADDRS:\t{faddrs:,}")
        logging.info(f"\tDEP:\t{fdep:,}")
        logging.info(f"\tPERIOD:\t{fperiod}")
        logging.info(f"\tBTC:\t{fbtc:,.8f}")
        logging.info(f"\tUSD:\t{fusd:,.2f}")
        logging.debug(f"\n{faddrs:,} & {fdep:,} & {fbtc:,.4f} & {fusd:,.0f}")


if __name__ == '__main__':
    desc = "Estimate Bitcoin revenue using different methodologies"
    version = '1.0'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-D', '--blocksci', type=str,
            help='Blocksci config file')
    parser.add_argument('-H', '--height', type=int, default=769900,
            help='Max block height')
    parser.add_argument('-s', '--seeds', action='append',
            help='Include this seed in the estimation')
    parser.add_argument('-f', '--seeds-file', default=None,
            help='File with a list of seeds to include in the estimation')
    parser.add_argument('-o', '--operation', type=str, default=None,
            help='Assign this tag to this operation')
    parser.add_argument('-A', '--artificial', action='store', default=None,
            help='Use this file to manually map addresses to new clusters')
    parser.add_argument('-C', '--clustering', action='store',
            choices=['mi', 'mica'], default=None,
            help='Use this expansion heuristic [mi:multi-input, mica:MI+CA]')
    parser.add_argument('-F', '--filter', action='store', default=None,
            choices=['vf', 'tf', 'vtf'], help=("Use CryptoLocker's value/time"
                "filter [vf:value, tf:time, vtf:vale&time]"))
    parser.add_argument('-T', '--tag-filter', action='store_true',
            help='Remove txes of addresses (except seeds) of service clusters')
    parser.add_argument('-d', '--doublecounting', action='store_true',
            help='Use the double-counting filter in the estimation')
    parser.add_argument('-O', '--output', type=str, default='./',
            help='Save all outputs into this folder')
    parser.add_argument('-v', '--version', action='version', version=version)

    args = parser.parse_args()
    e = set()
    if not (args.blocksci and os.path.isfile(args.blocksci)):
        e.add("Blocksci data not found.")
    if args.seeds_file and not os.path.isfile(args.seeds_file):
        e.add("A valid file should be given to the --seeds-file parameter.")
    if not args.seeds and not args.seeds_file:
        e.add("Please specify at least one seed using --seeds or --seeds-file")
    if args.artificial and not os.path.isfile(args.artificial):
        e.add("A valid file should be given to the --artificial parameter.")
    if args.output:
        if not os.path.isdir(args.output):
            e.add("A valid folder should be given to the --output parameter.")
        elif not os.access(args.output, os.W_OK):
            e.add(f"Folder without write permissions: {args.output}")
    if e:
        parser.error("\n".join(e))

    op = args.operation or 'estimation'
    est = estimation_name(args)
    flog = os.path.join(args.output, f"{op}_{est}.log")
    logging.basicConfig(filename=flog, level=logging.DEBUG)
    logging.debug(f"Estimation {est}. Version {version}")
    logging.debug(f"{args}")

    main(args)
