#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Module for converting BTCs to fiat currencies (e.g., USD) """

import os
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

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
    """
    Get minimum and maximum prices for ticker on day.
    """
    global ohlc_prices
    # Load data
    if ohlc_prices is None:
        d = pd.read_csv(historical_prices)
        d = d[d.crypto_name == ticker]
        ohlc_prices = {
            r['date']: {'high': r['high'], 'low': r['low']}
            for i, r in d.iterrows()
        }
    # Convert the day to string if it's a datetime object
    try:
        if not isinstance(day, str):
            day = datetime.strftime(day, '%Y-%m-%d')
    except ValueError:
        logging.info("Error: bad date format %s", day)
        return None
    # Return the entry in this day
    try:
        return ohlc_prices[day]
    except KeyError:
        msg = "Price on date {} not found in historical data, using"
        dday = datetime.strptime(day, '%Y-%m-%d')
        dday = dday.replace(tzinfo=timezone.utc)
        # Check if the date is before the first record
        if dday < datetime(2013, 5, 5, tzinfo=timezone.utc):
            logging.info("%s 2013-05-05", msg.format(day))
            return ohlc_prices['2013-05-05']
        # Check if the date is after the last record
        if dday > datetime(2022, 10, 23, tzinfo=timezone.utc):
            logging.info("%s 2022-10-23", msg.format(day))
            return ohlc_prices['2022-10-23']
        # Return the nearest day, going forward
        while True:
            dday += timedelta(days=1)
            next_day = datetime.strftime(dday, '%Y-%m-%d')
            if next_day in ohlc_prices:
                logging.info("%s %s", msg.format(day), next_day)
                return ohlc_prices[next_day]


def btc_to_currency_avg(amount, start, end=None, currency='USD'):
    """
    Convert BTC to the specific currency using an average period price using
    CoinDesk's Bitcoin Price Index API.
    :param amount: amount to convert in BTC
    :param start: starting date in the format YYYY-MM-DD
    :param end: ending date in the format YYYY-MM-DD
    :param currency: currency for the conversion in ISO-4217 format. Default=USD
    :return: Total USD corresponding to the amount in BTC. If only starting date
    is provided, the average price of such day will be used. The units are BTC.
    """
    try:
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        if not end:
            delta = timedelta(days=1)
            end_dt = start_dt
            start_dt = datetime.strptime(start, '%Y-%m-%d') - delta
        else:
            end_dt = datetime.strptime(end, '%Y-%m-%d')
    except ValueError:
        logging.info("Bad date format: (%s, %s)", start, end)
        return None
    start = datetime.strftime(start_dt, '%Y-%m-%d')
    end = datetime.strftime(end_dt, '%Y-%m-%d')
    ckey = (start, end, currency)
    if ckey in btc_price_cache:
        logging.debug("BTC price cache hit: %s", ckey)
        return btc_price_cache[ckey] * amount
    # Get the price matrix
    response = requests.get(bpi_api_url.format(*ckey))
    logging.debug("BPI Response: %s", response.status_code)
    # Check for errors
    price_json = response.json() if response.status_code == 200 else None
    if price_json:
        days = len(price_json['bpi'])
        avg = sum(price_json['bpi'].values()) / days
        btc_price_cache.update({ckey: avg})
        return avg * amount
    return None

def get_price(day, ticker='btc', currency='usd'):
    """
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
    """
    global prices
    if prices is None:
        # Load all prices to date from API
        end = datetime.strftime(datetime.now(), '%Y-%m-%d')
        # Search for the cached file
        fname = f"coingecko_{end}_{ticker}_{currency}.csv"
        fname = os.path.join(file_path, 'data', fname)
        if os.path.isfile(fname):
            logging.info("Reading cached prices from %s", fname)
            prices = read_cache_prices(fname)
        else:
            ckey = (cg_tickers[ticker], currency)
            response = requests.get(cg_api_url.format(*ckey))
            # Check for errors
            if response.status_code == 200:
                prices_matrix = response.json()['prices']
                prices = {
                    str(datetime.utcfromtimestamp(ts/1000))[:10]: p
                    for [ts, p] in prices_matrix
                }
                start = list(prices.keys())[0]
                msg = "Retreived %s prices for %s from %s to %s"
                logging.info(msg, currency, ticker, start, end)
                write_cache_prices(prices, fname)
            else:
                logging.info("API Error: status code %s", response.status_code)
                return 0.0
    try:
        if not isinstance(day, str):
            day = datetime.strftime(day, '%Y-%m-%d')
    except ValueError:
        logging.info("Error: bad date format %s", day)
        return 0.0

    try:
        return prices[day]
    except KeyError:
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
            logging.info("%s %s", msg.format(day), fday)
            return prices[fday]
        # Check if the date is after the last record
        if dday > lday:
            lday = str(lday)[:10]
            logging.info("%s %s", msg.format(day), lday)
            return prices[lday]
        # Return the nearest day, going forward
        while True:
            dday += timedelta(days=1)
            next_day = datetime.strftime(dday, '%Y-%m-%d')
            if next_day in prices:
                logging.info("%s %s", msg.format(day), next_day)
                return prices[next_day]

def get_price_coindesk(day, start='2010-07-17', currency='USD'):
    """
    Get the BTC price in the specific currency some date, using CoinDesk's
    Bitcoin Price Index API.

    :param day: Date string (YYYY-MM-DD) or datetime.datetime object with the
    date
    :param start: starting date in the format YYYY-MM-DD (default 2010-07-17).
    The price for older dates is default to 0.005
    :param currency: currency for the conversion in ISO-4217 format. Default=USD
    :return: The price of BTC in the given currency at the given date.
    :rtype: float
    """
    global cd_prices
    if cd_prices is None:
        # Load prices from API
        try:
            # If start is a datetime object, convert it to str
            if not isinstance(start, str):
                start = datetime.strftime(start, '%Y-%m-%d')
        except ValueError:
            logging.info("Error: bad start date format %s", start)
            return 0.0
        # Get all prices to date
        end = datetime.strftime(datetime.now(), '%Y-%m-%d')
        ckey = (start, end, currency)
        # Search for the cached file
        fname = f"coindesk_{start}_{end}_{currency}.csv"
        fname = os.path.join(file_path, 'data', fname)
        if os.path.isfile(fname):
            logging.info("Reading cached prices from %s", fname)
            cd_prices = read_cache_prices(fname)
        else:
            response = requests.get(bpi_api_url.format(*ckey))
            # Check for errors
            if response.status_code == 200:
                cd_prices = response.json()['bpi']
                write_cache_prices(cd_prices, fname)
                logging.info("Retreived cd_prices from %s to %s", start, end)
            else:
                logging.info("API Error: status code %s", response.status_code)
                return 0.0
    try:
        if not isinstance(day, str):
            day = datetime.strftime(day, '%Y-%m-%d')
    except ValueError:
        logging.info("Error: bad date format %s", day)
        return 0.0

    try:
        return cd_prices[day]
    except KeyError:
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
            logging.info("%s %s", msg.format(day), fday)
            return cd_prices[fday]
        # Check if the date is after the last record
        if dday > lday:
            lday = str(lday)[:10]
            logging.info("%s %s", msg.format(day), lday)
            return cd_prices[lday]
        # Return the nearest day, going forward
        while True:
            dday += timedelta(days=1)
            next_day = datetime.strftime(dday, '%Y-%m-%d')
            if next_day in cd_prices:
                logging.info("%s %s", msg.format(day), next_day)
                return cd_prices[next_day]

def write_cache_prices(w_prices, fname):
    """
    Write the cache of prices into a file.
    """
    d = {'date': [], 'price': []}
    for k, v in w_prices.items():
        d['date'].append(k)
        d['price'].append(v)

    df = pd.DataFrame(data=d)
    df.to_csv(fname, index=None)

def read_cache_prices(fname):
    """
    Read the cache of prices from a file.
    """
    d = pd.read_csv(fname)
    r_prices = {r.date: r.price for i, r in d.iterrows()}
    return r_prices

def get_price_coingecko(date, ticker='bch', currency='usd'):
    """
    Get the price of a crypto asset in the specific date, using CoinGecko's
    Price API.

    :param date: Date string (YYYY-MM-DD) or datetime.datetime object with the
    date
    :param ticker: Cryptocurrency asset (e.g. bch, ltc, btc). Default=bch
    :param currency: Currency for the conversion. Default=usd
    :return: The price of the crypto asset in the given currency at the given
    date
    :rtype: float
    """
    global cg_prices
    if cg_prices is None:
        # Load prices from API
        ticker = cg_tickers[ticker]
        response = requests.get(cg_api_url.format(ticker, currency))
        # Check for errors
        if response.status_code == 200:
            prices_matrix = response.json()['prices']
            cg_prices = {str(datetime.utcfromtimestamp(ts/1000))[:10]: p \
                    for [ts, p] in prices_matrix}
            logging.info("Retreived prices for %s in %s", ticker, currency)
        else:
            logging.info("API Error: status code %s", response.status_code)
            return 0.0
    try:
        if not isinstance(date, str):
            date = datetime.strftime(date, '%Y-%m-%d')
    except ValueError:
        logging.info("Error: bad date format %s", date)
        return 0.0

    try:
        return cg_prices[date]
    except KeyError:
        logging.info("Date not found in prices API (using 0.0): %s", date)
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
    today = datetime.now()
    logging.info("Today: %s", today)
    delta_1day = timedelta(days=1)
    today -= delta_1day
    logging.info("Last BTC value in USD (%s): %s", today, get_price(today))
    logging.info("BTC value in USD the 2010-01-01: %s", get_price('2010-01-01'))
    logging.info("BTC value in USD the 2019-09-30: %s", get_price('2019-09-30'))
    logging.info("BTC value in USD the 2023-05-18: %s", get_price('2023-05-18'))
    logging.info("BTC maximum price: %s", max(prices.values()))
    logging.info('-' * 50)
    logging.info("Historical BTC prices")
    hist_price = get_low_high_price('2013-01-01')
    logging.info("2013-01-01 high: %s", hist_price['high'])
    logging.info("2013-01-01 low:  %s", hist_price['low'])
    hist_price = get_low_high_price('2019-10-12')
    logging.info("2019-10-12 high: %s", hist_price['high'])
    logging.info("2019-10-12 low:  %s", hist_price['low'])
    hist_price = get_low_high_price('2023-01-01')
    logging.info("2023-01-01 high: %s", hist_price['high'])
    logging.info("2023-01-01 low:  %s", hist_price['low'])
