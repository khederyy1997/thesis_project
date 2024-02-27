import os

GROUP_IDs = {1: {"Left": 1, "Right": 2},
             2: {"Left": 4, "Right": 3},
             3: {"Left": 5, "Right": 6},
             4: {"Left": 7, "Right": 8},
             5: {"Left": 10, "Right": 9},
             7: {"Left": 13, "Right": 14},
             8: {"Left": 15, "Right": 16},
             9: {"Left": 18, "Right": 17},
             10: {"Left": 19, "Right": 20},
             11: {"Left": 21, "Right": 22},
             13: {"Left": 25, "Right": 26},
             24: {"Left": 47, "Right": 48},
             25: {"Left": 49, "Right": 50},
             26: {"Left": 51, "Right": 52},
             27: {"Left": 53, "Right": 54},
             28: {"Left": 55, "Right": 56},
             29: {"Left": 57, "Right": 58},
             30: {"Left": 59, "Right": 60}

            }

def flatten(ls): return [item for sublist in ls for item in sublist]

def annotationCompleted(rootdir, EXCLUDED):
    l = []
    a = os.walk(rootdir)
    for subdir, dirs, files in os.walk(rootdir):
        break
    for coppia in ["1", "2", "3", "4", "5", "7", "8", "9", "10", "11", "13"]:
        if coppia in EXCLUDED: continue
        couple = ((int(coppia)*2)-1, int(coppia)*2)
        for subdir, dirs, files in os.walk(rootdir):
            for id in couple:
                if f"{id}.csv" in files:
                    l.append(id)
                    break
    return l

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()