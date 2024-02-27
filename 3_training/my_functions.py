from sklearn.preprocessing import scale

def myRound(score, val = 3):
    var = str(round(score, val))
    tmp = "0" * ((val + 2) - len(var))
    var += tmp
    return var

# Train test split ma per la casistica "Leave one out"
def train_test_one_subject_out(df, id, isScale):
    train, test = df[df["id"] != id], df[df["id"] == id]

    #X_train = train.drop(["classe", "id", "inizio", "fine"], axis=1).copy()
    X_train = train.drop(["classe", "id"], axis=1).copy()

    y_train = train["classe"].copy()

    #X_test  = test.drop(["classe", "id", "inizio", "fine"], axis=1).copy()
    X_test  = test.drop(["classe", "id"], axis=1).copy()

    y_test  = test["classe"].copy()

    if isScale: X_train, X_test = scale(X_train), scale(X_test)
    return X_train, X_test, y_train, y_test


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