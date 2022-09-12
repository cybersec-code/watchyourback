#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import requests
import pandas as pd
from datetime import datetime

bcy_api_url = 'https://api.blockcypher.com/v1/btc/main/addrs/{}/meta'

bab_token = 'your_token_here'
bab_api_url = 'https://www.bitcoinabuse.com/api/reports/check'
bab_api_url += f'?api_token={bab_token}'
bab_api_url += '&address={}'

bab_api_all = 'https://www.bitcoinabuse.com/api/download/forever'
bab_api_all += f'?api_token={bab_token}'
abuse_types = {'ransomware': 'ransomware', 'scam': 'blackmail-scam',
        'sextortion': 'sextortion', 'tormarket': 'darknet-market',
        'mixer': 'bitcoin-tumbler', 'miscabuse': 'other'}

def all_reports(update=False):
    dest = '../data/tagging/bitcoinabuse/'
    if not update:
        return pd.read_csv(f"{dest}reports.csv")
    with requests.Session() as s:
        response = requests.get(bab_api_all)
        if 200 == response.status_code:
            print(f'Updating BitcoinAbuse reports file')
            response = requests.get(bab_api_all)
            # File is in ISO-8859-1 format, this escaped chars are problematic
            content = response.text.replace('\\"', '')
            data = list(csv.reader(content.splitlines(), delimiter=',',
                quotechar='"'))
            df = pd.DataFrame(data=data[1:], columns=data[0])
            # Save original file
            df = df.astype({'abuse_type_id': int})
            df.to_csv(f"{dest}reports.csv", index=None)
            # Stats of each address and abuse_type_id
            def lambda1(x): return (x==1).sum()   # ransomware
            def lambda2(x): return (x==2).sum()   # darknet-market
            def lambda3(x): return (x==3).sum()   # bitcoin-tumbler
            def lambda4(x): return (x==4).sum()   # blackmail-scam
            def lambda5(x): return (x==5).sum()   # sextortion
            def lambda99(x): return (x==99).sum() # other
            sdf = df.groupby('address').agg(
                    num_reports=('id', 'count'), #???
                    num_reports_ransomware=('abuse_type_id', lambda1),
                    num_reports_tormarket=('abuse_type_id', lambda2),
                    num_reports_mixer=('abuse_type_id', lambda3),
                    num_reports_scam=('abuse_type_id', lambda4),
                    num_reports_sextortion=('abuse_type_id', lambda5),
                    num_reports_miscabuse=('abuse_type_id', lambda99)
                )
            # Save stats
            d = datetime.now().strftime('%Y%m%d')
            sdf.to_csv(f"{dest}{d}_reports_stats.csv", index_label='address')
            # Majority voting for final tags
            mv = sdf.drop(['num_reports'], axis=1).idxmax(axis=1)
            tdf = pd.DataFrame(zip(mv.index,
                mv.apply(lambda x: f"{x.replace('num_reports_', '')}"),
                mv.apply(lambda x: f"bitcoinabuse_{abuse_types[x.replace('num_reports_', '')]}")),
                columns=['address', 'category', 'subtype'])
            tdf['url'] = 'https://www.bitcoinabuse.com'
            tdf['tag'] = ''
            tdf['category'] = 'miscabuse'
            # Save tags file
            tdf[['address', 'category', 'tag', 'subtype', 'url']].\
                    to_csv(f"{dest}{d}_bitcoinabuse.tags", index=None, header=None)
            return df
        else:
            sc = response.status_code
            print(f'Failed to update Bitcoin Abuse reports, status code: {sc}')
            return None

# Single address query for reports at blockchain abuse
def reports(addr):
    # TODO verify addr
    response = requests.get(bab_api_url.format(addr))
    print(f'BAB Response: {response.status_code}')
    # Check for errors
    report_json = response.json() if 200 == response.status_code else None
    return report_json

# Single address query for metadata at blockcypher
def metadata(addr):
    # TODO verify addr
    response = requests.get(bcy_api_url.format(addr))
    print(f'BCY Response: {response.status_code}')
    # Check for errors
    meta_info = response.json() if 200 == response.status_code else {}
    return meta_info

def parsed_metadata(addr='', info_json={}):
    if not addr and not info_json:
        return ''
    elif addr:
        info_json = metadata(addr)
    return "\n".join([f"{k[:20]}: {v[:20]}" for k, v in info_json.items()])

if __name__ == "__main__":
    # Testing blockcypher using an address from 'ViaBTC Bitcoin Mining Pool'
    addr = '18cBEMRxXHqzWWCxZNtU91F5sbUNKhL5PX'
    meta = metadata(addr)
    print('------------- with addr ---------------------------')
    print(parsed_metadata(addr))
    print('------------- with content ------------------------')
    print(parsed_metadata(info_json=meta))
    print('----------------------------------------')
    # Testing bitcoinabuse using a phishing address
    addr = '1N6dubqFmnyQ2qDWvi32ppVbc3kKMTYcGW'
    r = reports(addr)
    print(r)
    print('----------------------------------------')

