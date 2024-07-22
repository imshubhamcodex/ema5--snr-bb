from kiteconnectapp import *


def login(enctoken):  # Login return authorized obj
    return KiteApp(enctoken)

def get_instruments(kite, exchange):
    return kite.instruments(exchange)

def get_historical_data(kite, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
    return kite.historical_data(instrument_token, from_date, to_date, interval, continuous, oi)