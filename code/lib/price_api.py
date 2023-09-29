#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# CoinDesk API
bpi_api_url = 'https://api.coindesk.com/v1/bpi/historical/close.json'
bpi_api_url += '?start={}&end={}&currency={}'

# Save a cache of the BTC daily price
btc_price_cache = {}
prices = None
cd_prices = None

# CoinGecko API
cg_api_url = 'https://api.coingecko.com/api/v3/coins/{}/'
cg_api_url += 'market_chart?vs_currency={}&days=max&interval=daily'
cg_tickers = {'bch': 'bitcoin-cash', 'ltc': 'litecoin', 'btc': 'bitcoin'}
cg_prices = None

# Dataset obtained from Kaggle:
# https://www.kaggle.com/datasets/maharshipandya/
#       -cryptocurrency-historical-prices-dataset
file_path = os.path.abspath(os.path.dirname(__file__))
historical_prices = os.path.join(file_path, 'data', 'historical_prices.csv')
ohlc_prices = None


def get_low_high_price(day, ticker='Bitcoin'):
    global ohlc_prices
    # Load data
    if ohlc_prices is None:
        d = pd.read_csv(historical_prices)
        d = d[d.crypto_name==ticker]
        ohlc_prices = {
                        r['date']: {'high': r['high'], 'low': r['low']}
                        for i, r in d.iterrows()
                      }
    # Convert the day to string if it's a datetime object
    try:
        if not isinstance(day, str):
            day = datetime.strftime(day, '%Y-%m-%d')
    except ValueError as e:
        logging.info(f'Error: bad date format {day}')
        return None
    # Return the entry in this day
    try:
        return ohlc_prices[day]
    except KeyError as e:
        msg = "Price on date {} not found in historical data, using"
        dday = datetime.strptime(day, '%Y-%m-%d')
        dday = dday.replace(tzinfo=timezone.utc)
        # Check if the date is before the first record
        if dday < datetime(2013, 5, 5, tzinfo=timezone.utc):
            logging.info(f"{msg.format(day)} 2013-05-05")
            return ohlc_prices['2013-05-05']
        # Check if the date is after the last record
        elif dday > datetime(2022, 10, 23, tzinfo=timezone.utc):
            logging.info(f"{msg.format(day)} 2022-10-23")
            return ohlc_prices['2022-10-23']
        # Return the nearest day, going forward
        else:
            while True:
                dday += timedelta(days=1)
                next_day = datetime.strftime(dday, '%Y-%m-%d')
                if next_day in ohlc_prices:
                    logging.info(f"{msg.format(day)} {next_day}")
                    return ohlc_prices[next_day]


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

def get_price(day, ticker='btc', currency='usd'):
    '''
    Get the BTC price in the specific currency some date, using CoinGecko's
    Public API v3.

    :param day: Date in str format YYYY-MM-DD or datetime.datetime object.
    :param ticker: ticker of the crypto-currency (btc, bch, or ltc).
    :param currency: currency for the conversion (ISO-4217). Default is usd.
    :return: The price of the given crypto-currency in the given currency at
    the given day. Use the first or last date in the prices data if day is not
    within the data range, the closest day (forward) if the day is not found in
    the data, or 0.0 if some error occurs.
    :rtype: float
    '''
    global prices
    if prices is None:
        # Load all prices to date from API
        end = datetime.strftime(datetime.now(), '%Y-%m-%d')
        # Search for the cached file
        fname = f"coingecko_{end}_{ticker}_{currency}.csv"
        fname = os.path.join(file_path, 'data', fname)
        if os.path.isfile(fname):
            logging.info(f"Reading cached prices from {fname}")
            prices = read_cache_prices(fname)
        else:
            ckey = (cg_tickers[ticker], currency)
            response = requests.get(cg_api_url.format(*ckey))
            # Check for errors
            if 200 == response.status_code:
                prices_matrix = response.json()['prices']
                prices = {
                        str(datetime.utcfromtimestamp(ts/1000))[:10]: p
                        for [ts, p] in prices_matrix
                    }
                start = list(prices.keys())[0]
                msg = (
                        f"Retreived {currency} prices for {ticker} "
                        f"from {start} to {end}"
                    )
                logging.info(msg)
                write_cache_prices(prices, fname)
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
        msg = "Price on date {} not found, using"
        dday = datetime.strptime(day, '%Y-%m-%d')
        dday = dday.replace(tzinfo=timezone.utc)
        fday = datetime.strptime(list(prices.keys())[0], '%Y-%m-%d')
        fday = fday.replace(tzinfo=timezone.utc)
        lday = datetime.strptime(list(prices.keys())[-1], '%Y-%m-%d')
        lday = lday.replace(tzinfo=timezone.utc)
        # Check if the date is before the first record
        if dday < fday:
            # If its BTC, we fall-back to CoinDesk API
            if ticker == 'btc':
                return get_price_coindesk(day, currency=currency.upper())
            fday = str(fday)[:10]
            logging.info(f"{msg.format(day)} {fday}")
            return prices[fday]
        # Check if the date is after the last record
        elif dday > lday:
            lday = str(lday)[:10]
            logging.info(f"{msg.format(day)} {lday}")
            return prices[lday]
        # Return the nearest day, going forward
        else:
            while True:
                dday += timedelta(days=1)
                next_day = datetime.strftime(dday, '%Y-%m-%d')
                if next_day in prices:
                    logging.info(f"{msg.format(day)} {next_day}")
                    return prices[next_day]

def get_price_coindesk(day, start='2010-07-17', currency='USD'):
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
    global cd_prices
    if cd_prices is None:
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
        # Search for the cached file
        fname = f"coindesk_{start}_{end}_{currency}.csv"
        fname = os.path.join(file_path, 'data', fname)
        if os.path.isfile(fname):
            logging.info(f"Reading cached prices from {fname}")
            cd_prices = read_cache_prices(fname)
        else:
            response = requests.get(bpi_api_url.format(*ckey))
            # Check for errors
            if 200 == response.status_code:
                cd_prices = response.json()['bpi']
                write_cache_prices(cd_prices, fname)
                logging.info(f"Retreived cd_prices from {start} to {end}")
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
        return cd_prices[day]
    except KeyError as e:
        msg = "Price on date {} not found, using"
        dday = datetime.strptime(day, '%Y-%m-%d')
        dday = dday.replace(tzinfo=timezone.utc)
        fday = datetime.strptime(list(cd_prices.keys())[0], '%Y-%m-%d')
        fday = fday.replace(tzinfo=timezone.utc)
        lday = datetime.strptime(list(cd_prices.keys())[-1], '%Y-%m-%d')
        lday = lday.replace(tzinfo=timezone.utc)
        # Check if the date is before the first record
        if dday < fday:
            fday = str(fday)[:10]
            logging.info(f"{msg.format(day)} {fday}")
            return cd_prices[fday]
        # Check if the date is after the last record
        elif dday > lday:
            lday = str(lday)[:10]
            logging.info(f"{msg.format(day)} {lday}")
            return cd_prices[lday]
        # Return the nearest day, going forward
        else:
            while True:
                dday += timedelta(days=1)
                next_day = datetime.strftime(dday, '%Y-%m-%d')
                if next_day in cd_prices:
                    logging.info(f"{msg.format(day)} {next_day}")
                    return cd_prices[next_day]

def write_cache_prices(prices, fname):
    d = {'date': [], 'price': []}
    for k, v in prices.items():
        d['date'].append(k)
        d['price'].append(v)

    df = pd.DataFrame(data=d)
    df.to_csv(fname, index=None)

def read_cache_prices(fname):
    d = pd.read_csv(fname)
    prices = {r.date: r.price for i, r in d.iterrows()}
    return prices

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
    logging.info(f"BTC value in USD the 2010-01-01: {get_price('2010-01-01')}")
    logging.info(f"BTC value in USD the 2019-09-30: {get_price('2019-09-30')}")
    logging.info(f"BTC value in USD the 2023-05-18: {get_price('2023-05-18')}")
    logging.info(f"BTC maximum price: {max(prices.values())}")
    logging.info('-' * 50)
    logging.info(f"Historical BTC prices")
    hist_price = get_low_high_price('2013-01-01')
    logging.info(f"2013-01-01 high: {hist_price['high']}")
    logging.info(f"2013-01-01 low:  {hist_price['low']}")
    hist_price = get_low_high_price('2019-10-12')
    logging.info(f"2019-10-12 high: {hist_price['high']}")
    logging.info(f"2019-10-12 low:  {hist_price['low']}")
    hist_price = get_low_high_price('2023-01-01')
    logging.info(f"2023-01-01 high: {hist_price['high']}")
    logging.info(f"2023-01-01 low:  {hist_price['low']}")

