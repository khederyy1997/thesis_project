from globals import *
import pandas as pd
import os

""" DOC: File per la trasformazione della colonna dei secondi del file csv generato con ELAN;
(dovuta al fatto che la prima annotazione era stata presa con il numero dello screen e non con il timestamp) """

# Vado nella cartella superiore
path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
#print(os.getcwd())

folder, ID = 10, 20
COLUMNS = ["id", "classe", "inizio", "fine"]

ELAN = pd.read_csv(f"{folder}/{ID}ELAN.csv", header = None)
ELAN.drop(ELAN.columns[[0, 1]], axis = 1, inplace = True)

def toSec(timestamp): return round(int(timestamp) / 40) + 1

def toCSV(id, df):
    listone = [[id, row[2], toSec(row[0]), toSec(row[1])] for row in df.values]
    return pd.DataFrame(listone, columns = COLUMNS)

toCSV(ID, ELAN).to_csv(f"{folder}/{ID}.csv", index = False)

