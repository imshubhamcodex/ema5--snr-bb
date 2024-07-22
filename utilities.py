import os
import msvcrt
import time
from logger import printing
print = printing

def clear_screen():
    if os.name == "nt":
        _ = os.system("cls")
    else:
        _ = os.system("clear")


def read_from_file():  # Read data from auth.txt
    if os.path.exists("auth.txt"):
        with open("auth.txt", "r") as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                key, value = line.strip().split(": ")
                data[key] = value
        return data
    else:
        return None


def input_with_timeout(prompt, timeout):  # Taking input with expiration time
    print(prompt, end="", flush=True)
    start_time = time.time()
    input_chars = []

    while True:
        if msvcrt.kbhit():
            char = msvcrt.getch().decode("utf-8")
            input_chars.append(char)
            print(char, end="", flush=True)
            break

        if time.time() - start_time > timeout:
            print()
            return None

    print()
    return "".join(input_chars)


def get_login_creds():  # Input login creds
    user_id = input("Enter your user ID: ")
    password = input("Enter your password: ")
    enctoken = input("Enter your encryption token: ")
    return user_id, password, enctoken


def save_to_file(user_id, password, enctoken):  # Save login creds to auth.txt
    with open("auth.txt", "w") as f:
        f.write(f"user_id: {user_id}\n")
        f.write(f"password: {password}\n")
        f.write(f"enctoken: {enctoken}\n")


def findIToken(ticker, i_list):
    matched_instruments = []
    for instrument in i_list:
        if ticker.lower() in instrument['iName'].lower():
            matched_instruments.append(instrument)

    if matched_instruments:
        for instrument in matched_instruments:
            print(f"iToken: {instrument['iToken']}, iName: {instrument['iName']}")
    else:
        print(f"No instruments found matching '{ticker}'.")

    token_id = int(input("Enter iToken : "))

    for instrument in matched_instruments:
        if token_id == instrument['iToken']:
            # print(f"iToken: {instrument['iToken']}, iName: {instrument['iName']}")
            return token_id
    return 0
    


