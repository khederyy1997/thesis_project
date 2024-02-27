import pandas as pd
from sklearn.svm import SVC
import os
from pathlib import Path

os.chdir('../')

def posMax(array):

    tmp = 0
    index = None
    for i in range(len(array)):
        if array[i] > tmp:
            tmp = array[i]
            index = i
    
    return index

def dynamicGraph(predict, FPS):

    i = 0
    coda = []
    new_array = []
    counting = [0 for _ in range(0,5)]

    for _ in range(len(predict)):
        counting[predict[i]] += 1 # incremento di 1 la lista counting nella posizione del numero che leggo
        if len(coda) > FPS:
            counting[coda[0]] -= 1
            del coda[0]
        coda.append(predict[i])
        #print(coda)
        #print(f"{elem}-{elem + FPS}: {predict[i]} --> {posMax(counting)}")
        i += 1
        new_array.append(posMax(counting))
    
    return new_array

def getPrediction(model, FPS, id):

    print(f"Get Prediction {id} {FPS} fps")

    Path(f"0_data/predictions/").mkdir(parents=True, exist_ok=True)

    # train (annotazioni) e test (senza annotazioni)
    train = pd.read_csv(f"0_data/groundtruth/sequence/{FPS}fps.csv")
    test = pd.read_csv(f"0_data/{FPS}fps/{id}.csv")

    # Controllare che {id} non sia all'interno del train, e se c'è rimuoverlo
    train = train[train["id"] != id]

    train.drop(["id", "inizio", "fine"], axis = 1, inplace = True)

    # y_test non esiste perché non sono state effettuate le annotazioni
    X_train, y_train = train.drop("classe", axis = 1).copy(), train["classe"].copy()

    # Creo, alleno e vedo i risultati
    clf = model().fit(X_train, y_train)
    results = [-1 for x in range(FPS - 1)] + list(clf.predict(test))

    # Salvo i risultati in un file csv a due colonne [predict, dynamic]
    pd.DataFrame(results, columns = ["predict"]).to_csv(f"0_data/predictions/{id}_{FPS}fps.csv", index = False)

def baybstra(predict, len_seq):

    lista = []
    for i in range(len(predict)):
        start = (i+1) - (len_seq)
        if start < 0: start = 0
        ls = predict[start : i+1]
        weighted = [0 for _ in range(0, 5)]
        for j, x in enumerate(ls):
            weighted[x] += len(ls) - j
        result = posMax(weighted)
        lista.append(result)
        print(i, ls, weighted, result)

    return lista

if __name__ == "__main__":
    for fps in [20]:
        getPrediction(SVC, fps, 13)