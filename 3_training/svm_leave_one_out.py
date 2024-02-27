import pandas as pd
import sklearn
from numpy import shape
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import savefig
from statistics import mean
from my_functions import *
import json
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import os
import pickle


os.chdir('../0_data')

ACTION_COL = ["AU01_r", 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
              'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
STATS = []
R_FEATURES = [x + "_" + y for y in ACTION_COL for x in STATS]


def reports_to_txt(file_name, models_list, toScale=False, resampled=False, resampleTechnique=0):
    if resampleTechnique == 0:
        resampleTechniqueName = "ClusterCentroids"
    elif resampleTechnique == 1:
        resampleTechniqueName = "NearMiss"
    else:
        resampleTechniqueName = "RandomUnderSampler"

    listona = ["precision", "recall", "f1-score"]

    if not resampled:
        df = pd.read_csv(f"groundtruth/sequence/{file_name}_normalized.csv")
    else:
        df = pd.read_csv(f"groundtruth/sequence/resampled/{file_name}_normalized_{resampleTechniqueName}.csv")

    ids_list = df["id"].unique()
    print(ids_list)

    #QUANDO FACCIO LEAVE_1_SUBJECT_OUT USO ids_list
    #non ce numero "9" perche l'annotazione deve essere corretta
    ids_list = [1, 2, 3, 4, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    # ids_list = [1]
    print(ids_list)

    print("\n" + file_name)

    for model in models_list:

        # model name to pretty print
        model_name = str(model).split(".")[-1][:-2]
        print("--- " + model_name)
        lista = []
        tabellone = {}

        #SE VOGLIO TOGLIERE TUTTE LE MINs
        #X = np.array(df.drop(["classe", "id","min_AU01_r", "min_AU02_r", "min_AU04_r", "min_AU05_r", "min_AU06_r", "min_AU07_r", "min_AU09_r", "min_AU10_r", "min_AU12_r", "min_AU14_r", "min_AU15_r", "min_AU17_r", "min_AU20_r", "min_AU23_r", "min_AU25_r", "min_AU26_r",
        #"min_AU45_r"], axis=1))

        # questo for e per leaveonesubjectout
        for out in ids_list:

            print(f'******************* SUBJECT OUT: {out} ********************')

            tabella = {"precision": None, "recall": None, "f1-score": None, "accuracy": None}

            print(f"------- {out}")

            # leave one out
            X_train, X_test, y_train, y_test = train_test_one_subject_out(df, out, isScale = toScale)

            # classic train test
            #if not resampled:
            #    X_train, X_test, y_train, y_test = train_test_split(df.drop(R_FEATURES + ["classe", "id", "inizio", "fine"], axis=1), df["classe"], test_size=0.20)
            #else:
            #     X_train, X_test, y_train, y_test = train_test_split(df.drop(R_FEATURES + ["classe", "id"], axis=1), df["classe"], test_size=0.20)

            # Creo e alleno il modello
            # clf = model(random_state=42)

            parameters = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000, 10000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}

            # time window 25 - accuracy 0.65
            # parameters = {'kernel': ['rbf'], 'C': [10000], 'gamma': [0.00001]}

            # time window 20 - accuracy 0.64
            # parameters = {'kernel': ['rbf'], 'C': [100], 'gamma': [0.001]}

            # time window 25 - accuracy 0.71
            #parameters = {'kernel': ['rbf'], 'C': [10], 'gamma': [0.01]}

            clf = GridSearchCV(model, parameters, verbose=2, cv=5, scoring="f1_macro")

            # passare tutto il dataset (senza le colonne id, classe, inizio e fine da X)
            clf.fit(X_train, y_train)

            #pickle.dump(clf, open("../trained_model.sav", 'wb'))

            # Predict sul modello allenato
            y_true, y_pred = y_test, clf.predict(X_test)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_["mean_test_score"]
            stds = clf.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()

            report = classification_report(y_true, y_pred, output_dict=True)
            print(report)
            print()

            # Salvo i risultati in un file txt
            if int(out) // 10 == 0:
                out = f"0{out}"
            else:
                out = f'{out}'

            # Confusion Matrix
            array = confusion_matrix(y_true, y_pred)
            lista.append(array)
            print(array)
            # plot_confusion_matrix(clf, X_test, y_test)
            # savefig(f"confusion_matrix/{file_name}/ID{out}_{model_name}_{str(toScale)}_scale.png")

            tabella["accuracy"] = report["accuracy"]

            print("accuracy: ", report["accuracy"])

            for measure in listona:
                tabella[measure] = ([report[str(cls)][measure] for cls in range(1, 5)], report["macro avg"][measure],
                                    report["weighted avg"][measure])

                tabellone[str(out)] = tabella

        sum_array = ([[sum(matrix[i][j] for matrix in lista) for j in range(len(array[i]))] for i in range(len(array))])
        print(sum_array)
        df_cm = pd.DataFrame(sum_array)

        fig = plt.figure()
        sn.heatmap(df_cm, cmap="Blues")
        fig.savefig(f'a_average{model_name}.png', dpi=fig.dpi)

        with open(f'json/{file_name}_{model_name}_{str(toScale)}_scale.json', 'w') as write_file:
            json.dump(tabellone, write_file, indent=4)


Path("json/").mkdir(parents=True, exist_ok=True)

# reports_to_txt("20fps", [SVC], toScale = True)
reports_to_txt("50fps", [svm.SVC()], toScale=False, resampled=True, resampleTechnique=1)
