#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import requests
from datetime import datetime, timedelta

# CoinDesk API
bpi_api_url = 'https://api.coindesk.com/v1/bpi/historical/close.json'
bpi_api_url += '?start={}&end={}&currency={}'

# Save a cache of the BTC daily price
btc_price_cache = {}
prices = None

# CoinGecko API
cg_api_url = 'https://api.coingecko.com/api/v3/coins/{}/'
cg_api_url += 'market_chart?vs_currency={}&days=max&interval=daily'
cg_tickers = {'bch': 'bitcoin-cash', 'ltc': 'litecoin', 'btc': 'bitcoin'}
cg_prices = None

def btc_to_currency_avg(amount, start, end=None, currency='USD'):
    '''
    Convert BTC to the specific currency using an average period price using
    CoinDesk's Bitcoin Price Index API.
    :param amount: amount to convert in BTC
    :param start: starting date in the format YYYY-MM-DD
    :param end: ending date in the format YYYY-MM-DD
    :param currency: currency for the conversion in ISO-4217 format. Default=USD
    :return: Total USD corresponding to the amount in BTC. If only starting date
    is provided, the average price of such day will be used. The units are BTC.
    '''
    try:
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        if not end:
            delta = timedelta(days=1)
            end_dt = start_dt
            start_dt = datetime.strptime(start, '%Y-%m-%d') - delta
        else:
            end_dt = datetime.strptime(end, '%Y-%m-%d')
    except ValueError:
        logging.info(f'Bad dates format: {(start, end)}')
        return None
    start = datetime.strftime(start_dt, '%Y-%m-%d')
    end = datetime.strftime(end_dt, '%Y-%m-%d')
    ckey = (start, end, currency)
    if ckey in btc_price_cache:
        logging.debug(f'BTC price cache hit: {ckey}')
        return btc_price_cache[ckey] * amount
    else:
        response = requests.get(bpi_api_url.format(*ckey))
        logging.debug(f'BPI Response: {response.status_code}')
        # Check for errors
        price_json = response.json() if 200 == response.status_code else None
        if price_json:
            days = len(price_json['bpi'])
            avg = sum([x for x in price_json['bpi'].values()]) / days
            btc_price_cache.update({ckey: avg})
            return avg * amount
        else:
            return None

def get_price(day, start='2010-07-17', currency='USD'):
    '''
    Get the BTC price in the specific currency some date, using CoinDesk's
    Bitcoin Price Index API.

    :param day: Date string (YYYY-MM-DD) or datetime.datetime object with the
    date
    :param start: starting date in the format YYYY-MM-DD (default 2010-07-17).
    The price for older dates is default to 0.005
    :param currency: currency for the conversion in ISO-4217 format. Default=USD
    :return: The price of BTC in the given currency at the given date.
    :rtype: float
    '''
    global prices
    if prices is None:
        # Load prices from API
        try:
            # If start is a datetime object, convert it to str
            if not isinstance(start, str):
                start = datetime.strftime(start, '%Y-%m-%d')
        except ValueError:
            logging.info(f'Error: bad start date format {start}')
            return 0.0
        # Get all prices to date
        end = datetime.strftime(datetime.now(), '%Y-%m-%d')
        ckey = (start, end, currency)
        response = requests.get(bpi_api_url.format(*ckey))
        # Check for errors
        if 200 == response.status_code:
            prices = response.json()['bpi']
            logging.info(f"Retreived prices from {start} to {end}")
        else:
            logging.info(f'API Error: status code {response.status_code}')
            return 0.0
    try:
        if not isinstance(day, str):
            day = datetime.strftime(day, '%Y-%m-%d')
    except ValueError as e:
        logging.info(f'Error: bad date format {day}')
        return 0.0

    try:
        return prices[day]
    except KeyError as e:
        logging.info(f"Date not found in prices API (using 0.005): {day}")
        return 0.005

def get_price_coingecko(date, ticker='bch', currency='usd'):
    '''
    Get the price of a crypto asset in the specific date, using CoinGecko's
    Price API.

    :param date: Date string (YYYY-MM-DD) or datetime.datetime object with the
    date
    :param ticker: Cryptocurrency asset (e.g. bch, ltc, btc). Default=bch
    :param currency: Currency for the conversion. Default=usd
    :return: The price of the crypto asset in the given currency at the given
    date
    :rtype: float
    '''
    global cg_prices
    if cg_prices is None:
        # Load prices from API
        ticker = cg_tickers[ticker]
        response = requests.get(cg_api_url.format(ticker, currency))
        # Check for errors
        if 200 == response.status_code:
            prices_matrix = response.json()['prices']
            cg_prices = {str(datetime.utcfromtimestamp(ts/1000))[:10]: p \
                    for [ts, p] in prices_matrix}
            logging.info(f"Retreived prices for {ticker} in {currency}")
        else:
            logging.info(f'API Error: status code {response.status_code}')
            return 0.0
    try:
        if not isinstance(date, str):
            date = datetime.strftime(date, '%Y-%m-%d')
    except ValueError as e:
        logging.info(f'Error: bad date format {date}')
        return 0.0

    try:
        return cg_prices[date]
    except KeyError as e:
        logging.info(f"Date not found in prices API (using 0.0): {date}")
        return 0.0


if __name__ == "__main__":
#    # Test using a fixed date
#    today = '2019-09-30'
#    btc = btc_to_currency(1, today)
#    print(f"1 BTC == {btc} USD")
#    print('----------------------------------------')
#    # Test using some range
#    start = '2019-09-28'
#    end = '2019-09-30'
#    btc = btc_to_currency(1, start, end)
#    print(btc)
#    print('----------------------------------------')
#    # Test with a bad range
#    start = '2019-09-30'
#    end = '2019-09-28'
#    btc = btc_to_currency(1, start, end)
#    print(btc)
#    print('----------------------------------------')
#    # Test rising an exception
#    today = '2019-09-30 12:21'
#    btc = btc_to_currency(1, today)
#    print(btc)
#    print('----------------------------------------')
#    # Test the cache
#    today = '2019-09-30'
#    btc = btc_to_currency(2, today)
#    print(f"2 BTC == {btc} USD")
#    print('----------------------------------------')
#    print(btc_price_cache)
    logging.basicConfig(level=logging.DEBUG)
    day = datetime.now()
    logging.info(f"Today: {day}")
    delta = timedelta(days=1)
    day -= delta
    logging.info(f"Last BTC value in USD ({day}): {get_price(day)}")
    logging.info(f"BTC value in USD the 2019-09-30: {get_price('2019-09-30')}")
    logging.info(f"BTC maximum price: {max(prices.values())}")

