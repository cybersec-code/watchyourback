from datetime import datetime, timezone
from collections import defaultdict
from lib import price_api, blockchain_feature_extraction as bfe

#def is_cryptolocker_payment_date_relaxed(amount, fee, d):
#    # Convert amount and fee from satoshis to BTC
#    amount = amount / 1e8
#    fee = fee / 1e8
#    # 2 BTC between September 5, 2013 and November 11, 2013 allowing a
#    # three-day ransom period.
#    if amount == 2 or amount == (2 - fee):
#        return True
#    # 10 BTC between November 1, 2013 and November 11, 2013.
#    # The payment was the fee for using "CryptoLocker Decryption Service"
#    # that allowed victims, who failed to pay ransoms within the given time
#    # frame, to recover their files.
#    elif amount == 10 or amount == (10 - fee):
#        return True
#    # 1 BTC between November 8, 2013 and November 13, 2013 to allowing a
#    # three-day ransom period.
#    elif amount == 1 or amount == (1 - fee):
#        return True
#    # 0.5 BTC between November 10, 2013 and November 27, 2013 to allowing a
#    # three-day ransom period.
#    elif amount == 0.5 or amount == (0.5 - fee):
#        return True
#    # 2 BTC between November 11, 2013 and January 31, 2014. In this case, the
#    # payment was the reduced fee for using "CryptoLocker Decryption Service".
#    elif amount == 2 or amount == (2 - fee):
#        return True
#    # 0.3 BTC between November 24, 2013 and December 31, 2013.
#    elif amount == 0.3 or amount == (0.3 - fee):
#        return True
#    # 0.6 BTC between December 20, 2013 and January 31, 2014.
#    elif amount == 0.6 or amount == (0.6 - fee):
#        return True
#    else:
#        return False

def date(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)

def is_cryptolocker_payment_vtf(amount, fee, d):
    # Convert amount and fee from satoshis to BTC
    amount = amount / 1e8
    fee = fee / 1e8
    # Get price on specific date
    price = price_api.get_price(d)
    if price == 0.005:
        print(f"BTC to USD price on {d}: {price}")
    low_high_price = price_api.get_low_high_price(d)
    price_lo = low_high_price['low']  # price - price*0.01
    price_hi = low_high_price['high'] # price + price*0.01
    # Amount converted using low/high USD price
    vl = amount*price_lo
    vh = amount*price_hi
    fee_usd = fee*price
    # Convert date str to date object
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d')
        d = d.replace(tzinfo=timezone.utc)
    # 2 BTC between September 5, 2013 and November 11, 2013 allowing a
    # three-day ransom period.
    if d >= date(2013, 9, 5) and d < date(2013, 11, 12) \
        and (amount == 2 or amount == (2 - fee) \
            or (vl <= 2*price and 2*price <= vh) \
            or (vl <= 2*price-fee_usd and 2*price-fee_usd <= vh)):
        return '1) 2_BTC'
    # 10 BTC between November 1, 2013 and November 11, 2013.
    # The payment was the fee for using "CryptoLocker Decryption Service"
    # that allowed victims, who failed to pay ransoms within the given time
    # frame, to recover their files.
    elif d >= date(2013, 11, 1) and d < date(2013, 11, 12) \
        and (amount == 10 or amount == (10 - fee) \
            or (vl <= 10*price and 10*price <= vh) \
            or (vl <= 10*price-fee_usd and 10*price-fee_usd <= vh)):
        return '2) 10_BTC_late'
    # 1 BTC between November 8, 2013 and November 13, 2013 to allowing a
    # three-day ransom period.
    elif d >= date(2013, 11, 8) and d < date(2013, 11, 14) \
        and (amount == 1 or amount == (1 - fee) \
            or (vl <= 1*price and 1*price <= vh) \
            or (vl <= 1*price-fee_usd and 1*price-fee_usd <= vh)):
        return '3) 1_BTC'
    # 0.5 BTC between November 10, 2013 and November 27, 2013 to allowing a
    # three-day ransom period.
    elif d >= date(2013, 11, 10) and d < date(2013, 11, 28) \
        and (amount == 0.5 or amount == (0.5 - fee) \
            or (vl <= 0.5*price and 0.5*price <= vh) \
            or (vl <= 0.5*price-fee_usd and 0.5*price-fee_usd <= vh)):
        return '4) 0.5_BTC'
    # 2 BTC between November 11, 2013 and January 31, 2014. In this case, the
    # payment was the reduced fee for using "CryptoLocker Decryption Service".
    elif d >= date(2013, 11, 11) and d < date(2014, 2, 1) \
        and (amount == 2 or amount == (2 - fee) \
            or (vl <= 2*price and 2*price <= vh) \
            or (vl <= 2*price-fee_usd and 2*price-fee_usd <= vh)):
        return '5) 2_BTC_late'
    # 0.3 BTC between November 24, 2013 and December 31, 2013.
    elif d >= date(2013, 11, 24) and d < date(2014, 1, 1) \
        and (amount == 0.3 or amount == (0.3 - fee) \
            or (vl <= 0.3*price and 0.3*price <= vh) \
            or (vl <= 0.3*price-fee_usd and 0.3*price-fee_usd <= vh)):
        return '6) 0.3_BTC'
    # 0.6 BTC between December 20, 2013 and January 31, 2014.
    elif d >= date(2013, 12, 20) and d < date(2014, 2, 1) \
        and (amount == 0.6 or amount == (0.6 - fee) \
            or (vl <= 0.6*price and 0.6*price <= vh) \
            or (vl <= 0.6*price-fee_usd and 0.6*price-fee_usd <= vh)):
        return '7) 0.6_BTC'
    else:
        return None

def is_cryptolocker_payment_tf(amount, fee, d):
    # Convert date str to date object
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d')
        d = d.replace(tzinfo=timezone.utc)
    # 2 BTC between September 5, 2013 and November 11, 2013 allowing a
    # three-day ransom period.
    if d >= date(2013, 9, 5) and d < date(2013, 11, 12):
        return '1) 2_BTC'
    # 10 BTC between November 1, 2013 and November 11, 2013.
    # The payment was the fee for using "CryptoLocker Decryption Service"
    # that allowed victims, who failed to pay ransoms within the given time
    # frame, to recover their files.
    elif d >= date(2013, 11, 1) and d < date(2013, 11, 12):
        return '2) 10_BTC_late'
    # 1 BTC between November 8, 2013 and November 13, 2013 to allowing a
    # three-day ransom period.
    elif d >= date(2013, 11, 8) and d < date(2013, 11, 14):
        return '3) 1_BTC'
    # 0.5 BTC between November 10, 2013 and November 27, 2013 to allowing a
    # three-day ransom period.
    elif d >= date(2013, 11, 10) and d < date(2013, 11, 28):
        return '4) 0.5_BTC'
    # 2 BTC between November 11, 2013 and January 31, 2014. In this case, the
    # payment was the reduced fee for using "CryptoLocker Decryption Service".
    elif d >= date(2013, 11, 11) and d < date(2014, 2, 1):
        return '5) 2_BTC_late'
    # 0.3 BTC between November 24, 2013 and December 31, 2013.
    elif d >= date(2013, 11, 24) and d < date(2014, 1, 1):
        return '6) 0.3_BTC'
    # 0.6 BTC between December 20, 2013 and January 31, 2014.
    elif d >= date(2013, 12, 20) and d < date(2014, 2, 1):
        return '7) 0.6_BTC'
    else:
        return None

def is_cryptolocker_payment_vf(amount, fee, d):
    # Convert amount and fee from satoshis to BTC
    amount = amount / 1e8
    fee = fee / 1e8
    # Get price on specific date
    price = price_api.get_price(d)
    if price == 0.005:
        print(f"BTC to USD price on {d}: {price}")
    low_high_price = price_api.get_low_high_price(d)
    price_lo = low_high_price['low']  # price - price*0.01
    price_hi = low_high_price['high'] # price + price*0.01
    # Amount converted using low/high USD price
    vl = amount*price_lo
    vh = amount*price_hi
    fee_usd = fee*price
    # 2 BTC between September 5, 2013 and November 11, 2013 allowing a
    # three-day ransom period.
    if (amount == 2 or amount == (2 - fee) \
            or (vl <= 2*price and 2*price <= vh) \
            or (vl <= 2*price-fee_usd and 2*price-fee_usd <= vh)):
        return '1) 2_BTC'
    # 10 BTC between November 1, 2013 and November 11, 2013.
    # The payment was the fee for using "CryptoLocker Decryption Service"
    # that allowed victims, who failed to pay ransoms within the given time
    # frame, to recover their files.
    elif (amount == 10 or amount == (10 - fee) \
            or (vl <= 10*price and 10*price <= vh) \
            or (vl <= 10*price-fee_usd and 10*price-fee_usd <= vh)):
        return '2) 10_BTC_late'
    # 1 BTC between November 8, 2013 and November 13, 2013 to allowing a
    # three-day ransom period.
    elif (amount == 1 or amount == (1 - fee) \
            or (vl <= 1*price and 1*price <= vh) \
            or (vl <= 1*price-fee_usd and 1*price-fee_usd <= vh)):
        return '3) 1_BTC'
    # 0.5 BTC between November 10, 2013 and November 27, 2013 to allowing a
    # three-day ransom period.
    elif (amount == 0.5 or amount == (0.5 - fee) \
            or (vl <= 0.5*price and 0.5*price <= vh) \
            or (vl <= 0.5*price-fee_usd and 0.5*price-fee_usd <= vh)):
        return '4) 0.5_BTC'
    # 2 BTC between November 11, 2013 and January 31, 2014. In this case, the
    # payment was the reduced fee for using "CryptoLocker Decryption Service".
    elif (amount == 2 or amount == (2 - fee) \
            or (vl <= 2*price and 2*price <= vh) \
            or (vl <= 2*price-fee_usd and 2*price-fee_usd <= vh)):
        return '5) 2_BTC_late'
    # 0.3 BTC between November 24, 2013 and December 31, 2013.
    elif (amount == 0.3 or amount == (0.3 - fee) \
            or (vl <= 0.3*price and 0.3*price <= vh) \
            or (vl <= 0.3*price-fee_usd and 0.3*price-fee_usd <= vh)):
        return '6) 0.3_BTC'
    # 0.6 BTC between December 20, 2013 and January 31, 2014.
    elif (amount == 0.6 or amount == (0.6 - fee) \
            or (vl <= 0.6*price and 0.6*price <= vh) \
            or (vl <= 0.6*price-fee_usd and 0.6*price-fee_usd <= vh)):
        return '7) 0.6_BTC'
    else:
        return None

def is_cryptolocker_payment_btc(amount, fee, d):
    # Convert date str to date object
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d')
        d = d.replace(tzinfo=timezone.utc)
    # Convert amount and fee from satoshis to BTC
    amount = amount / 1e8
    fee = fee / 1e8
    # 2 BTC between September 5, 2013 and November 11, 2013 allowing a
    # three-day ransom period.
    if d >= date(2013, 9, 5) and d < date(2013, 11, 12) \
        and (amount == 2 or amount == (2 - fee)):
        return '1) 2_BTC'
    # 10 BTC between November 1, 2013 and November 11, 2013.
    # The payment was the fee for using "CryptoLocker Decryption Service"
    # that allowed victims, who failed to pay ransoms within the given time
    # frame, to recover their files.
    elif d >= date(2013, 11, 1) and d < date(2013, 11, 12) \
        and (amount == 10 or amount == (10 - fee)):
        return '2) 10_BTC_late'
    # 1 BTC between November 8, 2013 and November 13, 2013 to allowing a
    # three-day ransom period.
    elif d >= date(2013, 11, 8) and d < date(2013, 11, 14) \
        and (amount == 1 or amount == (1 - fee)):
        return '3) 1_BTC'
    # 0.5 BTC between November 10, 2013 and November 27, 2013 to allowing a
    # three-day ransom period.
    elif d >= date(2013, 11, 10) and d < date(2013, 11, 28) \
        and (amount == 0.5 or amount == (0.5 - fee)):
        return '4) 0.5_BTC'
    # 2 BTC between November 11, 2013 and January 31, 2014. In this case, the
    # payment was the reduced fee for using "CryptoLocker Decryption Service".
    elif d >= date(2013, 11, 11) and d < date(2014, 2, 1) \
        and (amount == 2 or amount == (2 - fee)):
        return '5) 2_BTC_late'
    # 0.3 BTC between November 24, 2013 and December 31, 2013.
    elif d >= date(2013, 11, 24) and d < date(2014, 1, 1) \
        and (amount == 0.3 or amount == (0.3 - fee)):
        return '6) 0.3_BTC'
    # 0.6 BTC between December 20, 2013 and January 31, 2014.
    elif d >= date(2013, 12, 20) and d < date(2014, 2, 1) \
        and (amount == 0.6 or amount == (0.6 - fee)):
        return '7) 0.6_BTC'
    else:
        return None

def cryptolocker_payments(addrs, chain):
    payments = defaultdict(list)
    for a in addrs:
        a = bfe.addr_from_string(a, chain)
        for o in a.outputs.to_list():
            ts = datetime.utcfromtimestamp(o.tx.block.timestamp)
            ts = ts.replace(tzinfo=timezone.utc)
            payment = is_cryptolocker_payment_vtf(o.value, o.tx.fee, ts)
#            payment = is_cryptolocker_payment_btc(o.value, o.tx.fee, ts)
            if payment:
                hash = str(o.tx.hash)
                ts = str(ts)
                payments[payment].append((hash, o.index, o.value/1e8, ts))
    return payments

