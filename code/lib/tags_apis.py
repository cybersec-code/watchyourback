#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import requests
import pandas as pd
from datetime import datetime

bcy_api_url = 'https://api.blockcypher.com/v1/btc/main/addrs/{}/meta'

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

