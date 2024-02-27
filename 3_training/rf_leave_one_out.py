import pandas as pd
import sklearn
from numpy import shape
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


def recall1(label4, confusion_matrix2):
    row = confusion_matrix2[label4, :]
    return confusion_matrix2[label4, label4] / row.sum()


def precision1(label5, confusion_matrix3):
    col = confusion_matrix3[:, label5]
    return confusion_matrix3[label5, label5] / col.sum()


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

    #if not resampled:
    df = pd.read_csv(f"groundtruth/sequence/{file_name}_normalized.csv")
    #else:
    #    df = pd.read_csv(f"groundtruth/sequence/resampled/{file_name}_normalized_{resampleTechniqueName}.csv")

    ids_list = df["id"].unique()
    print(ids_list)

    #QUANDO FACCIO LEAVE_1_SUBJECT_OUT USO ids_list
    #non ce numero "9" perche l'annotazione deve essere corretta
    ids_list = [1, 2, 3, 4, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    #50fps muore su 15 quindi rimosso e 18
    #ids_list = [1, 2, 3, 4, 7, 8, 10, 13, 14, 16, 17, 19, 20, 21, 22]

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

            # *************** da qui per stimare i parametri:
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=300, stop=3000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 150, num=11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            random_grid = {'n_estimators': n_estimators, \
                           'max_features': max_features, \
                           'max_depth': max_depth, \
                           'min_samples_split': min_samples_split, \
                           'min_samples_leaf': min_samples_leaf, \
                           'bootstrap': bootstrap}

            rf = RandomForestClassifier()
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, cv=3, verbose=2,
                                           n_jobs=3)

            rf_random.fit(X_train, y_train)

            y_true, y_pred = y_test, rf_random.predict(X_test)

            print("Best parameters set found on development set:")
            print()
            print(rf_random.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = rf_random.cv_results_["mean_test_score"]
            stds = rf_random.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, rf_random.cv_results_["params"]):
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

        mean_of_conf_matrix_arrays = np.sum(lista, axis=0)

        print("Cofusion marix")
        print(mean_of_conf_matrix_arrays)

        print("Accuracy")
        diagonal_sum = mean_of_conf_matrix_arrays.trace()
        sum_of_all_elements = mean_of_conf_matrix_arrays.sum()
        print(diagonal_sum / sum_of_all_elements)

        print("label precision recall")
        # print(f"{precision('1', mean_of_conf_matrix_arrays):9.3f} {recall('1', mean_of_conf_matrix_arrays):6.3f}")

        print(recall1(0, mean_of_conf_matrix_arrays))
        print(recall1(1, mean_of_conf_matrix_arrays))
        print(recall1(2, mean_of_conf_matrix_arrays))
        print(recall1(3, mean_of_conf_matrix_arrays))
        print(precision1(0, mean_of_conf_matrix_arrays))
        print(precision1(1, mean_of_conf_matrix_arrays))
        print(precision1(2, mean_of_conf_matrix_arrays))
        print(precision1(3, mean_of_conf_matrix_arrays))

        sumf1 = 0
        sumf2 = 0
        sumf3 = 0
        sumf4 = 0
        sumpr = 0
        sumrc = 0

        print("f1")
        print((2.0 * precision1(0, mean_of_conf_matrix_arrays) * recall1(0, mean_of_conf_matrix_arrays)) / (
                precision1(0, mean_of_conf_matrix_arrays) + recall1(0, mean_of_conf_matrix_arrays)))

        sumf1 = (2.0 * precision1(0, mean_of_conf_matrix_arrays) * recall1(0, mean_of_conf_matrix_arrays)) / (
                precision1(0, mean_of_conf_matrix_arrays) + recall1(0, mean_of_conf_matrix_arrays))

        sumpr = precision1(0, mean_of_conf_matrix_arrays)
        sumrc = recall1(0, mean_of_conf_matrix_arrays)

        print((2.0 * precision1(1, mean_of_conf_matrix_arrays) * recall1(1, mean_of_conf_matrix_arrays)) / (
                precision1(1, mean_of_conf_matrix_arrays) + recall1(1, mean_of_conf_matrix_arrays)))

        sumf2 = (2.0 * precision1(1, mean_of_conf_matrix_arrays) * recall1(1, mean_of_conf_matrix_arrays)) / (
                precision1(1, mean_of_conf_matrix_arrays) + recall1(1, mean_of_conf_matrix_arrays))

        sumpr += precision1(1, mean_of_conf_matrix_arrays)
        sumrc += recall1(1, mean_of_conf_matrix_arrays)

        print((2.0 * precision1(2, mean_of_conf_matrix_arrays) * recall1(2, mean_of_conf_matrix_arrays)) / (
                precision1(2, mean_of_conf_matrix_arrays) + recall1(2, mean_of_conf_matrix_arrays)))

        sumf3 = (2.0 * precision1(2, mean_of_conf_matrix_arrays) * recall1(2, mean_of_conf_matrix_arrays)) / (
                precision1(2, mean_of_conf_matrix_arrays) + recall1(2, mean_of_conf_matrix_arrays))

        sumpr += precision1(2, mean_of_conf_matrix_arrays)
        sumrc += recall1(2, mean_of_conf_matrix_arrays)

        print((2.0 * precision1(3, mean_of_conf_matrix_arrays) * recall1(3, mean_of_conf_matrix_arrays)) / (
                precision1(3, mean_of_conf_matrix_arrays) + recall1(3, mean_of_conf_matrix_arrays)))

        sumf4 = (2.0 * precision1(3, mean_of_conf_matrix_arrays) * recall1(3, mean_of_conf_matrix_arrays)) / (
                precision1(3, mean_of_conf_matrix_arrays) + recall1(3, mean_of_conf_matrix_arrays))

        sumpr += precision1(3, mean_of_conf_matrix_arrays)
        sumrc += recall1(3, mean_of_conf_matrix_arrays)

        print("macro Fscore")
        print((sumf1 + sumf2 + sumf3 + sumf4) / 4)

        print("macro Precision")
        print(sumpr / 4)

        print("macro Recall")
        print(sumrc / 4)

        row1 = 0
        row2 = 0
        row3 = 0
        row4 = 0

        row1 = mean_of_conf_matrix_arrays[0, :]
        row2 = mean_of_conf_matrix_arrays[1, :]
        row3 = mean_of_conf_matrix_arrays[2, :]
        row4 = mean_of_conf_matrix_arrays[3, :]

        print("weighted Fscore")
        print((sumf1 * row1.sum() + sumf2 * row2.sum() + sumf3 * row3.sum() + sumf4 * row4.sum()) / (
                row1.sum() + row2.sum() + row3.sum() + row4.sum()))


        fig = plt.figure()
        sn.heatmap(df_cm, cmap="Blues")
        fig.savefig(f'a_average{model_name}.png', dpi=fig.dpi)

        with open(f'json/{file_name}_{model_name}_{str(toScale)}_scale.json', 'w') as write_file:
            json.dump(tabellone, write_file, indent=4)


Path("json/").mkdir(parents=True, exist_ok=True)

# reports_to_txt("20fps", [SVC], toScale = True)
reports_to_txt("25fps", [svm.SVC()], toScale=False, resampled=False, resampleTechnique=1)
