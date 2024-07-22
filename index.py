import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

from dateutil.tz import tzoffset

from colorama import init, Fore, Back, Style
from invoker import (
    prepare_data,
    login_with_enctoken,
    get_instruments_list,
    get_historical_dataset
)
from utilities import findIToken
from logger import printing
print = printing

def calculate_bollinger_bands(df, length=20, multiplier=2.0):
    df['basis'] = df['close'].rolling(window=length).mean()
    df['dev'] = df['close'].rolling(window=length).std() * multiplier
    df['upperBB'] = df['basis'] + df['dev']
    df['lowerBB'] = df['basis'] - df['dev']
    return df

def calculate_ema(df, length=5):
    df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
    return df

def calculate_rsi(df, length=14):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_support_resistance(df, lookback):
    df['support'] = df['low'].rolling(window=lookback).min()
    df['resistance'] = df['high'].rolling(window=lookback).max()
    return df

def get_default_lookback(timeframe):
    if timeframe == "60":
        return 75
    elif timeframe == "15":
        return 80
    elif timeframe == "5":
        return 50
    elif timeframe == "3":
        return 70
    elif timeframe == "1":
        return 450
    else:
        return 75

def check_conditions(df):
    valid_candles = []

    prev_valid = False  # Flag to track if the previous candle was valid

    for i in range(1, len(df)):
        if (df['marked_l'].iloc[i] or df['marked_h'].iloc[i]) and (df['rsi'].iloc[i-1] >= 70 or df['rsi'].iloc[i-1] <= 30):
            if df['high'].iloc[i-1] >= df['upperBB'].iloc[i-1] or df['low'].iloc[i-1] <= df['lowerBB'].iloc[i-1]:
                if df['marked_l'].iloc[i] and abs(df['high'].iloc[i-1] - df['resistance'].iloc[i-1]) <= 20:
                    if not prev_valid:
                        valid_candles.append(df.index[i-1])
                    prev_valid = True
                elif df['marked_h'].iloc[i] and abs(df['low'].iloc[i-1] - df['support'].iloc[i-1]) <= 20:
                    if not prev_valid:
                        valid_candles.append(df.index[i-1])
                    prev_valid = True
                else:
                    prev_valid = False
            else:
                prev_valid = False
        else:
            prev_valid = False

    return valid_candles

def valid_candles_plot(df, timeframe):
    df.drop('volume', axis=1, inplace=True)

    # Define your parameters
    lengthBB = 20
    multBB = 2.0
    user_lookback = 0
    lengthEMA = 5
    lengthRSI = 14

    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df, lengthBB, multBB)

    # Determine lookback period
    default_lookback = get_default_lookback(timeframe)
    lookback = user_lookback if user_lookback > 0 else default_lookback

    # Calculate support and resistance levels
    df = calculate_support_resistance(df, lookback)

    # Calculate EMA
    df = calculate_ema(df, lengthEMA)

    # Calculate RSI
    df = calculate_rsi(df, lengthRSI)


    # Initialize columns for marked candles
    df['marked_l'] = False
    df['marked_low'] = np.nan
    df['marked_h'] = False
    df['marked_high'] = np.nan

    # Check conditions for marking candles
    df['ema_above'] = (df['open'] > df['ema']) & (df['low'] > df['ema'])
    df['ema_below'] = (df['open'] < df['ema']) & (df['high'] < df['ema'])

    # Loop through the DataFrame to mark candles
    for i in range(1, len(df)):
        if df['ema_above'].iloc[i-1]:
            df.at[i, 'marked_l'] = True
            df.at[i, 'marked_low'] = df['low'].iloc[i-1]
        if df['ema_below'].iloc[i-1]:
            df.at[i, 'marked_h'] = True
            df.at[i, 'marked_high'] = df['high'].iloc[i-1]

        if df['marked_l'].iloc[i] and df['low'].iloc[i] < df['marked_low'].iloc[i]:
            df.at[i, 'marked_l'] = True
            df.at[i, 'marked_low'] = df['low'].iloc[i]
        else:
            df.at[i, 'marked_l'] = False
            df.at[i, 'marked_low'] = np.nan

        if df['marked_h'].iloc[i] and df['high'].iloc[i] > df['marked_high'].iloc[i]:
            df.at[i, 'marked_h'] = True
            df.at[i, 'marked_high'] = df['high'].iloc[i]
        else:
            df.at[i, 'marked_h'] = False
            df.at[i, 'marked_high'] = np.nan

    # Check the specified conditions using the separate function
    valid_candles = check_conditions(df)

    # Mark the valid candles in the DataFrame
    df['valid_candle'] = df.index.isin(valid_candles)
    df['valid_candle_marker'] = np.where(df['valid_candle'], df['low'] - 10, np.nan)  # Position the marker slightly below the low

    # Plotting with mplfinance
    add_plots = [
        mpf.make_addplot(df['basis'], color='purple', linestyle='-', width=1.0, panel=0),
        mpf.make_addplot(df['upperBB'], color='red', linestyle='-', width=1.0, panel=0),
        mpf.make_addplot(df['lowerBB'], color='red', linestyle='-', width=1.0, panel=0),
        mpf.make_addplot(df['ema'], color='blue', linestyle='-', width=1.0, panel=0),
        mpf.make_addplot(df['support'], color='green', linestyle='-', width=1.0, panel=0),
        mpf.make_addplot(df['resistance'], color='orange', linestyle='-', width=1.0, panel=0),
        mpf.make_addplot(df['marked_low'], type='scatter', markersize=25, marker='v', color='red', panel=0),
        mpf.make_addplot(df['marked_high'], type='scatter', markersize=25, marker='^', color='green', panel=0),
        mpf.make_addplot(df['valid_candle_marker'], type='scatter', markersize=50, marker='o', color='blue', panel=0),  # Marker for valid candles
        mpf.make_addplot(df['rsi'], panel=1, color='black', title='RSI')  # Add RSI to the plot on a separate panel
    ]

    # Ensure index is of type DatetimeIndex
    df.index = pd.to_datetime(df.index)
    mpf.plot(df, type='candle', style='charles', addplot=add_plots, title='Candlestick Chart with Indicators', volume=False)
    df.reset_index(drop=True, inplace=True)

    return valid_candles

def reverse_trade_down(df, tp_ratio, sl_offset, trade_details, index_j):
    j = index_j
    tp = df['close'].iloc[j] - tp_ratio*abs(df['high'].iloc[j] - df['low'].iloc[j])
    sl = df['high'].iloc[j] + sl_offset

    for k in range(j + 1, len(df)):
        if df['high'].iloc[k] >= sl:
            trade = {
                "loss" : abs(sl - df['close'].iloc[j]),
                "profit": 0.0,
                "trade": "down",
                "date": df['date'].iloc[j]
            }
            trade_details.append(trade)
            print(df['date'].iloc[j])
            print(df['date'].iloc[k])
            print("---------------------s*")
            break
        elif df['low'].iloc[k] <= tp:
            trade = {
                "loss" : 0.0,
                "profit": abs(tp - df['close'].iloc[j]),
                "trade": "down",
                "date": df['date'].iloc[j]
            }
            trade_details.append(trade)
            break

    return trade_details

def reverse_trade_up(df, tp_ratio, sl_offset, trade_details, index_j):
    j = index_j
    tp = df['close'].iloc[j] + tp_ratio*abs(df['high'].iloc[j] - df['low'].iloc[j])
    sl = df['low'].iloc[j] - sl_offset

    for k in range(j + 1, len(df)):
        if df['low'].iloc[k] <= sl:
            trade = {
                "loss" : abs(sl - df['close'].iloc[j]),
                "profit": 0.0,
                "trade": "up",
                "date": df['date'].iloc[j]
            }
            trade_details.append(trade)
            print(df['date'].iloc[j])
            print(df['date'].iloc[k])
            print("---------------------s^")
            break
        elif df['high'].iloc[k] >= tp:
            trade = {
                "loss" : 0.0,
                "profit": abs(tp - df['close'].iloc[j]),
                "trade": "up",
                "date": df['date'].iloc[j]
            }
            trade_details.append(trade)
            break

    return trade_details

def main():
    # Login data fetch
    user_id, password, enctoken = prepare_data()

    # Trying login
    kite = login_with_enctoken(enctoken)

    print(Back.GREEN + "Hi", kite.profile()["user_shortname"],", successfully logged in." + Style.RESET_ALL)
    print(" ")

    # Fetching instruments list
    i_list = get_instruments_list(kite)

    timeframe = input("Enter Timeframe: ")
    ticker = input("Enter instrument name: ")
    iToken = findIToken(ticker, i_list)
    # iToken = "256265" # nifty 50 

    if not iToken:
        print("Invalid iToken")
        print("Exiting")
        return
    
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=1*180)
    delta_days = 100     # download date at once, in days
    interval = timeframe + "minute"

    print(f"\n{iToken} -- Downloading data...\n")
    historical_data = get_historical_dataset(kite, iToken, start_date, end_date, interval, delta_days)

    df = pd.DataFrame(historical_data)

    valid_candles = valid_candles_plot(df, timeframe)

    trade_details = []
    tp_ratio = 2

    if timeframe == "60":
        tp_ratio = 1.4
    elif timeframe == "15":
        tp_ratio = 2.5
    elif timeframe == "5":
        tp_ratio = 4
    elif timeframe == "3":
        tp_ratio = 3
    else:
        tp_ratio = 3

    sl_offset = 0
    with_rev_trade = False

    for i in valid_candles:

        from datetime import time
        start_time = time(9, 30)
        end_time = time(15, 0)
        current_time = df['date'].iloc[i].time()

        if start_time <= current_time <= end_time:
            print("Good time", current_time)
        else:
            print("Bad time", current_time)
            continue


        for j in range(i + 1, len(df)):
            if df['marked_h'].iloc[i + 1]:
                tp = df['high'].iloc[i] + tp_ratio*abs(df['high'].iloc[i] - df['low'].iloc[i])
                sl = df['low'].iloc[i] - sl_offset

                if df['low'].iloc[j] <= sl:
                    trade = {
                        "loss" : abs(sl - df['high'].iloc[i]),
                        "profit": 0.0,
                        "trade": "up",
                        "date": df['date'].iloc[i]
                    }
                    print(df['date'].iloc[i].time())
                    print(df['date'].iloc[j])
                    print("---------------------^")
                    trade_details.append(trade)
                    if with_rev_trade:
                        trade_details = reverse_trade_down(df, tp_ratio, sl_offset, trade_details, j)
                    break
                
                elif df['high'].iloc[j] >= tp:
                    trade = {
                        "loss" : 0.0,
                        "profit": abs(tp - df['high'].iloc[i]),
                        "trade": "up",
                        "date": df['date'].iloc[i]
                    }
                    trade_details.append(trade)
                    break
                    
            elif df['marked_l'].iloc[i + 1]:
                tp = df['low'].iloc[i] - tp_ratio*abs(df['high'].iloc[i] - df['low'].iloc[i])
                sl = df['high'].iloc[i] + sl_offset

                if df['high'].iloc[j] >= sl:
                    trade = {
                        "loss" : abs(sl - df['low'].iloc[i]),
                        "profit": 0.0,
                        "trade": "down",
                        "date": df['date'].iloc[i]
                    }
                    print(df['date'].iloc[i])
                    print(df['date'].iloc[j])
                    print("---------------------*")
                    trade_details.append(trade)
                    if with_rev_trade:
                        trade_details = reverse_trade_up(df, tp_ratio, sl_offset, trade_details, j)
                    break

                elif df['low'].iloc[j] <= tp:
                    trade = {
                        "loss" : 0.0,
                        "profit": abs(tp - df['low'].iloc[i]),
                        "trade": "down",
                        "date": df['date'].iloc[i]
                    }
                    trade_details.append(trade)
                    break
    
    profit_up = 0
    loss_up = 0

    profit_down = 0
    loss_down = 0

    wins_up = 0
    losses_up = 0

    wins_down = 0
    losses_down = 0

    # Process trade details
    for trade in trade_details:
        if trade['trade'] == 'up':
            if trade['profit'] > 0:
                profit_up += trade['profit']
                wins_up += 1
            else:
                loss_up += trade['loss']
                losses_up += 1
        elif trade['trade'] == 'down':
            if trade['profit'] > 0:
                profit_down += trade['profit']
                wins_down += 1
            else:
                loss_down += trade['loss']
                losses_down += 1

    # Print or return the results

    print(f"Profit Up: {round(profit_up, 2)}")
    print(f"Loss Up: {round(loss_up, 2)}")
    print(f"Wins Up: {wins_up}")
    print(f"Losses Up: {losses_up}")
    print("--------------------------------------------")
    print(f"Profit Down: {round(profit_down, 2)}")
    print(f"Loss Down: {round(loss_down, 2)}")
    print(f"Wins Down: {wins_down}")
    print(f"Losses Down: {losses_down}")
    print("--------------------------------------------")
    print(f"Total Profit: {round(profit_up + profit_down, 2)}")
    print(f"Total Loss: {round(loss_up + loss_down, 2)}")
    print(f"Overall: {round(profit_up + profit_down - loss_up - loss_down, 2)}")
    print(f"Wins: {(wins_up + wins_down)} , Loss:  {(losses_up + losses_down)}")
    print(f"Total trade: {wins_up + wins_down + losses_up + losses_down}")

if __name__ == "__main__":
    init()  # For colorama
    print(Back.RED)
    print("Designed by Shubham @https://github.com/imshubhamcodex/Kite")
    print(Style.RESET_ALL)
    main()
