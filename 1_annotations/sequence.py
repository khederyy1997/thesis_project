import pandas as pd
from statistics import mean, stdev
# from scipy.stats import skew, kurtosis
from numpy import mean as np_mean, std, absolute, percentile, square, sqrt
from globals import printProgressBar

""" DOC: Creazione dei file csv *sequenze* (unione tra l'output di openface e le annotazioni)"""

BASIC = ["id", "classe", "inizio", "fine"]
STATS = ["mean", "stdev", "mad", "iqr", "en", "rms"]

# len_seq: numero della lunghezza della sequenza di frame
# csv: nome del file csv da leggere in input

def make_csv_of_sequences(len_seq, file_name_csv):

    csv = pd.read_csv(file_name_csv)
    df = csv.values

    FEATURES = [x + "_" + y for y in csv.columns[4:] for x in STATS]
    COLUMNS = BASIC + FEATURES # colonne del nuovo csv

    listone = []

    # per la print progress bar
    total = len(df)
    printProgressBar(0, total, prefix = f"{len_seq}fps")

    i = 0
    discarded = 0

    while i < len(df):
        row = df[i]
        id      = int(row[0])
        start   = int(row[2])
        end     = int(row[3])

        if end - start >= len_seq - 1:
            classe = int(df[i][1]) # classe della sequenza
            df_seq = df[i:i + len_seq] # sub_df della sequenza

            if classe == 5:
                i = i + 1
                continue # skippo classe 5

            ls_seq = [id, classe, start, start + len_seq - 1]

            for feature in range(4,21): # per ogni feature
                arr_seq = [df_seq[x][feature] for x in range(len(df_seq))] # sequenza della singola feature

                stats = [
                    mean(arr_seq), 
                    stdev(arr_seq),
                    np_mean(absolute(arr_seq - np_mean(arr_seq))),
                    percentile(arr_seq, 75) - percentile(arr_seq, 25),
                    sum(square(arr_seq)),
                    sqrt(np_mean(square(arr_seq)))
                ]

                stats = [round(x, 4) for x in stats]
                
                for col in stats:
                    ls_seq.append(col)

            listone.append(ls_seq)

            i = i + len_seq

        else:
            i = i + (end - start) + 1
            discarded = discarded + 1

        printProgressBar(i+1, total, prefix=f"{len_seq}fps")

    print(f'for {len_seq} I generated {len(listone)} and discarded {discarded} ({(discarded/(discarded + len(listone))) * 100}%) sequences')

    return pd.DataFrame(listone, columns=COLUMNS)
