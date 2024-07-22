import datetime

outputFile = open('./logs/' +(datetime.datetime.now()).strftime("%Y-%m-%d %H_%M_%S") + ".log", 'w')
def printing(*args, end='\n', flush=False):
    # Print to console
    print(*args, end=end, flush=flush)
    # Print to file
    for arg in args:
        outputFile.write(str(arg) + '\n')