from datetime import datetime
import time
import CONSTANT
from colorama import Back, Style
from logger import printing
print = printing


def prepare_data():  # Preparing login auth data
    user_id, password, enctoken = None, None, None
    existing_data = read_from_file()

    if existing_data:
        print("User Exist: ")
        user_id = existing_data["user_id"]
        password = existing_data["password"]
        enctoken = existing_data["enctoken"]
        print("Your User ID: ", user_id)
        print("Your Password: ", password)
        print("Your Enctoken: ", enctoken)

        user_input = input_with_timeout("Overwrite this data? (y/n) (defalut n): ", 5)
        if user_input is None or user_input.lower() != "y":
            print("Proceeding without change.")
            print(" ")
            return user_id, password, enctoken

    print("Please overwrite the login creds.")
    print(" ")

    user_id, password, enctoken = get_login_creds()
    save_to_file(user_id, password, enctoken)
    print("Proceeding with changes.")
    return user_id, password, enctoken


def login_with_enctoken(enctoken):  # Login return authorized obj
    return login(enctoken)


def get_instruments_list(kite):
    i_list = []
    exchange = CONSTANT.EXCHANGE_NSE
    instruments = get_instruments(kite, exchange)

    for instrument in instruments:
        if instrument['name']:
            instrument_name = instrument['name'] 
            instrument_token = instrument['instrument_token']
            instrument_data = {
                "iToken" : instrument_token,
                "iName" : instrument_name
            }
            i_list.append(instrument_data)
    return i_list


def get_historical_dataset(kite, iToken, from_date, to_date, interval, delta_days=100):
    import datetime
    instrument_token = iToken
    delta = datetime.timedelta(days=delta_days)
    data = []

    while from_date < to_date:
        # Set the time to 09:15:00 for from_date and end_date
        from_date_with_time = from_date.replace(hour=9, minute=15, second=0)
        end_date_with_time = (from_date + delta).replace(hour=9, minute=15, second=0)
        
        if end_date_with_time > to_date:
            end_date_with_time = to_date.replace(hour=9, minute=15, second=0)
        
        # Format the dates to the required format
        from_date_str = str(from_date_with_time.strftime('%Y-%m-%d %H:%M:%S'))
        end_date_str = str(end_date_with_time.strftime('%Y-%m-%d %H:%M:%S'))
       
        data += get_historical_data(kite, instrument_token, from_date_str, end_date_str, interval)
        time.sleep(1)
        
        from_date = end_date_with_time + datetime.timedelta(days=1)

    return data

#     # 3minutes -> 4 months
#     # 5minutes -> 4 months
#     # 15minutes -> 7 months









from utilities import read_from_file, input_with_timeout, get_login_creds, save_to_file
from client import login, get_instruments, get_historical_data
